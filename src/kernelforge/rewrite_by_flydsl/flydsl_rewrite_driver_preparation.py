# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author or repair a rewrite-specific dual-path measurement driver.

This module deliberately does not depend on ``loop.task_preparer``.  A rewrite
driver has a different contract and lifecycle: it owns a source reference path,
a not-yet-implemented FlyDSL candidate path, and two independently timed modes.
Keeping the preparation engine here prevents either contract from silently
changing the other.

The agent works in an isolated temporary git repository containing read-only
copies of the task evidence.  Only one self-contained driver file can be
published.  The caller's source tree and candidate are therefore never writable
during preparation, and the destination driver is replaced only while the
deterministic rewrite contract is being checked.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
    with_writable_sandbox,
)
from kernelforge.agent_backends.registry import create_registered_backend
from kernelforge.config import Config
from kernelforge.resources import resource_path
from kernelforge.rewrite_by_flydsl import driver_contract, protocol
from kernelforge.rewrite_by_flydsl.budget import DEFAULT_REWRITE_BUDGET
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec
from kernelforge.durable_io import atomic_write_bytes


DRIVER_PREPARATION_FAILED = "driver_preparation_failed"
DRIVER_PREPARATION_DEADLINE = "driver_preparation_deadline"
INVOCATION_SPEC_INVALID = "invocation_spec_invalid"

DEFAULT_MAX_ATTEMPTS = 3
MAX_EVIDENCE_BYTES = 1024 * 1024

_EVIDENCE_DIR = "evidence"
_SOURCE_EVIDENCE = "source_kernel.py"
_CANDIDATE_EVIDENCE = "candidate_skeleton.py"
_INVOCATION_EVIDENCE = "invocation_spec.json"
_SOFTMAX_REFERENCE = "reference_softmax_driver.py"
_MXFP8_REFERENCE = "reference_mxfp8_grouped_gemm_driver.py"


@dataclass
class DriverPreflight:
    """One complete pre-PORT check of the rewrite driver contract."""

    report: driver_contract.PreflightReport
    reference: driver_contract.PreflightReport | None = None
    candidate_probe: driver_contract.PreflightReport | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def failure_class(self) -> str:
        return self.report.failure_class

    @property
    def detail(self) -> str:
        return self.report.detail

    @property
    def source_ms(self) -> float | None:
        return self.reference.timing_ms if self.reference is not None else None

    @property
    def reference_case_ids(self) -> tuple[str, ...]:
        return self.reference.case_ids if self.reference is not None else ()


@dataclass
class DriverPreparationResult:
    """Explicit result of the isolated rewrite driver preparation."""

    ok: bool
    attempts: int = 0
    preflight: DriverPreflight | None = None
    failure_class: str = ""
    error: str = ""
    audit_dir: str = ""
    wrote_driver: bool = False


def _failed_preflight(failure_class: str, detail: str) -> DriverPreflight:
    return DriverPreflight(
        report=driver_contract.PreflightReport(
            ok=False,
            failure_class=failure_class,
            detail=detail,
        )
    )


def _remaining(deadline_unix: float | None) -> float:
    if not deadline_unix or deadline_unix <= 0:
        return float("inf")
    return deadline_unix - time.time()


def _stage_timeout(ceiling: int, deadline_unix: float | None) -> int:
    remaining = _remaining(deadline_unix)
    if remaining == float("inf"):
        return ceiling
    return max(1, min(ceiling, int(remaining)))


def preflight_rewrite_driver(
    spec: RewriteSpec,
    driver_path: str,
    *,
    deadline_unix: float | None = None,
) -> DriverPreflight:
    """Run the complete deterministic contract required before PORT starts."""

    independence = driver_contract.check_driver_independence(spec, driver_path)
    if not independence.ok:
        return DriverPreflight(report=independence)

    remaining = _remaining(deadline_unix)
    if remaining <= 0:
        return _failed_preflight(
            DRIVER_PREPARATION_DEADLINE,
            "the rewrite deadline was reached before the source driver preflight",
        )
    reference = driver_contract.preflight_reference(
        spec,
        driver_path,
        timeout_sec=_stage_timeout(
            DEFAULT_REWRITE_BUDGET.reference_preflight_timeout_sec,
            deadline_unix,
        ),
    )
    if not reference.ok:
        return DriverPreflight(
            report=reference,
            reference=reference,
            warnings=list(reference.warnings),
        )

    remaining = _remaining(deadline_unix)
    if remaining <= 0:
        return DriverPreflight(
            report=driver_contract.PreflightReport(
                ok=False,
                failure_class=DRIVER_PREPARATION_DEADLINE,
                detail=("the rewrite deadline was reached before the candidate argument probe"),
            ),
            reference=reference,
            warnings=list(reference.warnings),
        )
    candidate_probe = driver_contract.probe_candidate_arguments(
        spec,
        driver_path,
        timeout_sec=_stage_timeout(
            DEFAULT_REWRITE_BUDGET.candidate_probe_timeout_sec,
            deadline_unix,
        ),
    )
    warnings = [*reference.warnings, *candidate_probe.warnings]
    if not candidate_probe.ok:
        return DriverPreflight(
            report=candidate_probe,
            reference=reference,
            candidate_probe=candidate_probe,
            warnings=warnings,
        )
    return DriverPreflight(
        report=driver_contract.PreflightReport(ok=True),
        reference=reference,
        candidate_probe=candidate_probe,
        warnings=warnings,
    )


def _read_evidence(path: Path, *, required: bool = True) -> bytes:
    try:
        data = path.read_bytes()
    except OSError:
        if required:
            raise
        return b""
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError(f"preparation evidence exceeds {MAX_EVIDENCE_BYTES} bytes: {path}")
    return data


def _load_invocation_spec(path: str) -> bytes:
    if not path:
        return b""
    source = Path(path).resolve()
    data = _read_evidence(source)
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invocation spec is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("invocation spec must contain a JSON object")
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def _reference_examples() -> dict[str, bytes]:
    examples = {
        _SOFTMAX_REFERENCE: resource_path("examples/triton2flydsl-softmax-flydsl-rewrite/driver.py"),
        _MXFP8_REFERENCE: resource_path("examples/triton2flydsl-mxfp8-grouped-gemm/driver.py"),
    }
    return {name: content for name, path in examples.items() if (content := _read_evidence(path, required=False))}


def _write_evidence(
    stage: Path,
    spec: RewriteSpec,
    invocation_spec: bytes,
) -> dict[Path, tuple[bytes, int]]:
    evidence_dir = stage / _EVIDENCE_DIR
    evidence_dir.mkdir()
    payloads = {
        _SOURCE_EVIDENCE: _read_evidence(Path(spec.source_kernel)),
        _CANDIDATE_EVIDENCE: _read_evidence(Path(spec.flydsl_kernel)),
        **_reference_examples(),
    }
    if invocation_spec:
        payloads[_INVOCATION_EVIDENCE] = invocation_spec

    snapshots: dict[Path, tuple[bytes, int]] = {}
    for name, payload in payloads.items():
        path = evidence_dir / name
        path.write_bytes(payload)
        path.chmod(0o444)
        snapshots[path] = (payload, 0o444)
    return snapshots


def _restore_evidence(snapshots: dict[Path, tuple[bytes, int]]) -> None:
    for directory in {path.parent for path in snapshots}:
        if directory.is_symlink():
            directory.unlink()
        directory.mkdir(parents=True, exist_ok=True)
    for path, (content, mode) in snapshots.items():
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            path.chmod(0o644)
        path.write_bytes(content)
        path.chmod(mode)


def _unexpected_stage_outputs(
    stage: Path,
    stage_driver: Path,
    evidence_paths: set[Path],
) -> list[Path]:
    allowed = {stage_driver.resolve(), *(path.resolve() for path in evidence_paths)}
    unexpected: list[Path] = []
    for path in stage.rglob("*"):
        if ".git" in path.parts:
            continue
        if not path.is_file() and not path.is_symlink():
            continue
        if path.resolve() not in allowed:
            unexpected.append(path)
    return unexpected


def _clean_unexpected_outputs(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            continue


def _ensure_agent_git_workspace(stage: Path) -> None:
    commands = [
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c",
            "user.name=KernelForge",
            "-c",
            "user.email=kernel-forge@localhost",
            "commit",
            "--allow-empty",
            "-qm",
            "rewrite driver preparation baseline",
        ],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            cwd=stage,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"could not initialize the isolated driver preparation workspace: {result.stderr or result.stdout}"
            )


async def _run_agent(
    *,
    config: Config,
    stage: Path,
    stage_driver: Path,
    evidence_paths: set[Path],
    prompt: str,
    timeout_sec: int,
    progress_log: list[str],
) -> str:
    runtime = with_writable_sandbox(config.agent_runtime())
    backend = create_registered_backend(runtime)
    run_spec = AgentRunSpec(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=prompt,
        cwd=str(stage),
        writable=True,
        timeout_sec=timeout_sec,
        target_files=[str(stage_driver)],
        # Deliberately no driver_script, for the reason task_preparer records at
        # its own AgentRunSpec: that field declares the measurement surface the
        # guard must defend, so it snapshots the driver as protected. Here the
        # driver is the artifact being authored, and naming it both target and
        # protected made every attempt end in
        #   "protected tracked files changed: <driver>"
        # followed by a rollback to the placeholder -- the agent wrote a working
        # driver, verified it in --ref-bench-mode, and had the file reverted out
        # from under it three times in a row. target_files already carries it.
        protected_globs=[path.name for path in evidence_paths],
        allow_dirty_targets=True,
        allow_untracked=False,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=True,
            max_turns=config.max_turns,
            permission_mode=os.environ.get("FORGE_PERMISSION_MODE", "acceptEdits"),
            bare=False,
        ),
        progress_log=progress_log,
    )
    result = await asyncio.wait_for(
        backend.run(run_spec),
        timeout=watchdog_timeout_sec(timeout_sec),
    )
    return result.text.strip()


def _audit_root(experiments_dir: str, operator_slug: str) -> Path:
    root = Path(experiments_dir).resolve() / "rewrite_driver_preparation"
    root.mkdir(parents=True, exist_ok=True)
    audit = root / (f"{operator_slug}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}")
    audit.mkdir()
    return audit


def _audit_text(audit: Path, relative: str, text: str) -> None:
    try:
        path = audit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass


def _audit_json(audit: Path, relative: str, payload: dict) -> None:
    try:
        _audit_text(
            audit,
            relative,
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        )
    except (TypeError, ValueError):
        pass


def _audit_driver(audit: Path, relative: str, driver: Path) -> None:
    try:
        if driver.is_file():
            destination = audit / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(driver, destination)
    except OSError:
        pass


def _preflight_payload(preflight: DriverPreflight) -> dict:
    return {
        "report": asdict(preflight.report),
        "reference": (asdict(preflight.reference) if preflight.reference is not None else None),
        "candidate_probe": (asdict(preflight.candidate_probe) if preflight.candidate_probe is not None else None),
        "warnings": list(preflight.warnings),
    }


def _restore_driver(path: Path, original: bytes | None) -> None:
    """Put the caller's own driver back, permissions included."""
    if original is None:
        path.unlink(missing_ok=True)
        return
    atomic_write_bytes(path, original)


def _build_prompt(
    *,
    spec: RewriteSpec,
    stage_driver: Path,
    invocation_spec_available: bool,
    prior_failure: str,
) -> str:
    shape_text = json.dumps(spec.shapes, indent=2, sort_keys=True, default=str)
    invocation_note = (
        f"Read `{_EVIDENCE_DIR}/{_INVOCATION_EVIDENCE}` first. It contains the "
        "call evidence supplied by the orchestrator."
        if invocation_spec_available
        else (
            "No invocation spec was supplied. Derive only facts justified by the "
            "source and task metadata; do not invent integer, routing, mask, or "
            "quantization-scale domains."
        )
    )
    retry = f"\n## Deterministic failure from the previous attempt\n{prior_failure}\n" if prior_failure else ""
    return f"""\
Create or repair the self-contained rewrite measurement driver at
`{stage_driver.name}`.

## Task
- logical operation: `{spec.op_name}`
- source entry hint: `{spec.source_entry or "(not supplied)"}`
- source target functions: {spec.target_functions or "(not supplied)"}
- required FlyDSL builder symbol: `{spec.builder_symbol}`
- source path at runtime: environment variable
  `{protocol.ENV_SOURCE_KERNEL}`
- candidate path at runtime: environment variable
  `{protocol.ENV_CANDIDATE_KERNEL}`
- builder symbol at runtime: environment variable
  `{protocol.ENV_BUILDER_SYMBOL}`
- logical operation at runtime: environment variable
  `{protocol.ENV_LOGICAL_OP}`
- task shapes:
```json
{shape_text}
```

## Read-only evidence
Read `{_EVIDENCE_DIR}/{_SOURCE_EVIDENCE}` and
`{_EVIDENCE_DIR}/{_CANDIDATE_EVIDENCE}`. The two reference drivers under
`{_EVIDENCE_DIR}/` demonstrate the protocol, but they are examples rather than
operator semantics. {invocation_note}

## Required driver behavior
1. Default mode runs the complete correctness suite. It constructs semantically
   valid inputs, invokes the original source implementation and the FlyDSL
   candidate on identical cases, then prints `SNR: <value> dB` and/or
   `allclose: True`.
2. `--ref-bench-mode` times only the source implementation. It prints
   `case_ms: <case_id> <ms>` for every case and one `median_ms: <ms>` aggregate.
3. `--bench-mode` times only the FlyDSL candidate and prints the same case ids
   and timing keys. Candidate loading must be lazy: while the supplied candidate
   is still a skeleton this mode must fail without printing a timing.
4. Accept `--warmup` and `--iters`. Also implement `--profile-run` as a
   candidate-only invocation without reference work or timing output so the
   later optimizer can profile the driver without rewriting it.
5. Load source and candidate from the producer-owned absolute paths above.
   Add the source directory to `sys.path` before executing the source module so
   its local imports remain valid. Do not rely on a hard-coded module name.
6. Use deterministic, domain-correct inputs. In particular, never create index,
   routing, mask, packed FP8, or exponent-scale tensors with
   `torch.randn(...).to(integer_or_fp8_dtype)`.
7. The driver must not import KernelForge and must not call one implementation
   from the other implementation's timing path.

## Write boundary
Modify only `{stage_driver.name}`. The finished driver must be one
self-contained Python file. Do not edit evidence and do not create helper,
configuration, cache, or generated files. Save the best complete driver before
running optional checks.
{retry}
"""


_SYSTEM_PROMPT = """\
You are responsible only for authoring a measurement driver for a source-kernel
to FlyDSL rewrite. The driver is an executable correctness oracle and benchmark,
not the kernel implementation. Preserve source/candidate isolation, use valid
operator inputs, and implement the complete dual-path stdout contract. You work
inside an isolated staging repository: edit only the requested driver file.
"""


async def prepare_rewrite_driver(
    *,
    spec: RewriteSpec,
    driver_path: str,
    config: Config,
    experiments_dir: str,
    deadline_unix: float | None,
    invocation_spec_file: str = "",
    initial_preflight: DriverPreflight | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> DriverPreparationResult:
    """Author or repair one driver without exposing the caller's tree to writes."""

    destination = Path(driver_path).resolve()
    if destination in {
        Path(spec.source_kernel).resolve(),
        Path(spec.flydsl_kernel).resolve(),
    }:
        return DriverPreparationResult(
            ok=False,
            failure_class=driver_contract.DRIVER_NOT_INDEPENDENT,
            error="the driver destination collides with a kernel under validation",
        )
    if "forge_experiments" in destination.parts:
        return DriverPreparationResult(
            ok=False,
            failure_class=driver_contract.DRIVER_NOT_INDEPENDENT,
            error=("the driver destination is producer-owned experiment state and cannot own the correctness gate"),
        )
    try:
        invocation_spec = _load_invocation_spec(invocation_spec_file)
    except (OSError, ValueError) as error:
        return DriverPreparationResult(
            ok=False,
            failure_class=INVOCATION_SPEC_INVALID,
            error=str(error),
        )

    try:
        audit = _audit_root(experiments_dir, spec.operator_slug)
        original = destination.read_bytes() if destination.is_file() else None
        original_mode = destination.stat().st_mode & 0o777 if destination.is_file() else 0o644
    except OSError as error:
        return DriverPreparationResult(
            ok=False,
            failure_class=DRIVER_PREPARATION_FAILED,
            error=f"could not initialize rewrite driver preparation: {error}",
        )
    last_preflight = initial_preflight
    prior_failure = initial_preflight.detail if initial_preflight is not None and not initial_preflight.ok else ""
    if initial_preflight is not None:
        _audit_json(audit, "initial_preflight.json", _preflight_payload(initial_preflight))

    attempts_run = 0
    try:
        with tempfile.TemporaryDirectory(prefix="kernel_forge_rewrite_driver_") as temporary:
            stage = Path(temporary)
            stage_driver = stage / destination.name
            if original is None:
                stage_driver.write_text(
                    '"""Rewrite measurement driver; prepared by KernelForge."""\n',
                    encoding="utf-8",
                )
            else:
                stage_driver.write_bytes(original)
                stage_driver.chmod(original_mode)
            evidence = _write_evidence(stage, spec, invocation_spec)
            _ensure_agent_git_workspace(stage)

            for attempt in range(1, max(1, max_attempts) + 1):
                remaining = _remaining(deadline_unix)
                preflight_reserve = DEFAULT_REWRITE_BUDGET.driver_preflight_reserve_sec
                if remaining <= preflight_reserve:
                    break
                attempts_run = attempt
                timeout_sec = max(
                    1,
                    int(
                        min(
                            float(config.agent_timeout_sec),
                            remaining - preflight_reserve,
                        )
                    ),
                )
                prompt = _build_prompt(
                    spec=spec,
                    stage_driver=stage_driver,
                    invocation_spec_available=bool(invocation_spec),
                    prior_failure=prior_failure,
                )
                attempt_dir = f"attempt_{attempt:02d}"
                _audit_text(audit, f"{attempt_dir}/prompt.md", prompt)
                _audit_text(audit, f"{attempt_dir}/system_prompt.md", _SYSTEM_PROMPT)
                _audit_driver(audit, f"{attempt_dir}/driver_before.py", stage_driver)
                progress_log: list[str] = []
                try:
                    output = await _run_agent(
                        config=config,
                        stage=stage,
                        stage_driver=stage_driver,
                        evidence_paths=set(evidence),
                        prompt=prompt,
                        timeout_sec=timeout_sec,
                        progress_log=progress_log,
                    )
                    _audit_text(audit, f"{attempt_dir}/agent_output.txt", output)
                except asyncio.TimeoutError:
                    prior_failure = (
                        f"the previous authoring session timed out after {timeout_sec}s; save a complete driver earlier"
                    )
                    _audit_json(
                        audit,
                        f"{attempt_dir}/agent_event.json",
                        {"status": "timeout", "timeout_sec": timeout_sec},
                    )
                except Exception as error:  # noqa: BLE001
                    prior_failure = f"agent invocation failed: {type(error).__name__}: {error}"
                    _audit_json(
                        audit,
                        f"{attempt_dir}/agent_event.json",
                        {
                            "status": "error",
                            "type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                finally:
                    _audit_text(
                        audit,
                        f"{attempt_dir}/agent_progress.txt",
                        "\n".join(progress_log),
                    )
                    _audit_driver(
                        audit,
                        f"{attempt_dir}/driver_after_agent.py",
                        stage_driver,
                    )

                unexpected = _unexpected_stage_outputs(
                    stage,
                    stage_driver,
                    set(evidence),
                )
                evidence_changed = [
                    path
                    for path, (content, _mode) in evidence.items()
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.parent.is_symlink()
                        or path.read_bytes() != content
                    )
                ]
                _restore_evidence(evidence)
                _clean_unexpected_outputs(unexpected)
                if unexpected or evidence_changed:
                    changed = [
                        *(path.relative_to(stage).as_posix() for path in unexpected),
                        *(path.relative_to(stage).as_posix() for path in evidence_changed),
                    ]
                    prior_failure = (
                        "the previous attempt violated the write boundary: "
                        + ", ".join(sorted(set(changed)))
                        + "; modify only the driver"
                    )
                    continue
                if not stage_driver.is_file() or stage_driver.is_symlink():
                    prior_failure = "the previous attempt deleted the required driver"
                    continue

                candidate_bytes = stage_driver.read_bytes()
                try:
                    compile(
                        candidate_bytes,
                        str(destination),
                        "exec",
                        dont_inherit=True,
                    )
                except (SyntaxError, ValueError) as error:
                    prior_failure = f"the generated driver is not valid Python: {error}"
                    continue

                atomic_write_bytes(destination, candidate_bytes)
                try:
                    last_preflight = preflight_rewrite_driver(
                        spec,
                        str(destination),
                        deadline_unix=deadline_unix,
                    )
                except Exception as error:  # noqa: BLE001
                    _restore_driver(destination, original)
                    prior_failure = f"deterministic preflight raised {type(error).__name__}: {error}"
                    continue
                _audit_driver(
                    audit,
                    f"{attempt_dir}/driver_at_preflight.py",
                    destination,
                )
                _audit_json(
                    audit,
                    f"{attempt_dir}/preflight.json",
                    _preflight_payload(last_preflight),
                )
                if last_preflight.ok:
                    return DriverPreparationResult(
                        ok=True,
                        attempts=attempt,
                        preflight=last_preflight,
                        audit_dir=str(audit),
                        wrote_driver=True,
                    )
                _restore_driver(destination, original)
                prior_failure = f"[{last_preflight.failure_class}] {last_preflight.detail}"
    except (OSError, RuntimeError, ValueError) as error:
        _restore_driver(destination, original)
        return DriverPreparationResult(
            ok=False,
            attempts=attempts_run,
            preflight=last_preflight,
            failure_class=DRIVER_PREPARATION_FAILED,
            error=f"could not prepare the isolated driver workspace: {error}",
            audit_dir=str(audit),
        )

    _restore_driver(destination, original)
    deadline_reached = _remaining(deadline_unix) <= DEFAULT_REWRITE_BUDGET.driver_preflight_reserve_sec
    detail = prior_failure or "the preparation agent produced no conforming driver"
    if deadline_reached:
        detail = f"driver preparation reached its deadline; last failure: {detail}"
    return DriverPreparationResult(
        ok=False,
        attempts=attempts_run,
        preflight=last_preflight,
        failure_class=(DRIVER_PREPARATION_DEADLINE if deadline_reached else DRIVER_PREPARATION_FAILED),
        error=detail,
        audit_dir=str(audit),
    )
