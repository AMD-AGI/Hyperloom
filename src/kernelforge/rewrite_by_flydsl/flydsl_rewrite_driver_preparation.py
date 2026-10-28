# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author or repair a rewrite-specific dual-path measurement driver."""

from __future__ import annotations

import asyncio
import json
import logging
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
from kernelforge.agent_backends.runtime import agent_runtime
from kernelforge.config import Config
from kernelforge.resources import resource_path
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS
from kernelforge.rewrite_by_flydsl import driver_contract, protocol
from kernelforge.rewrite_by_flydsl.budget import DEFAULT_REWRITE_BUDGET
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec
from kernelforge.tools.durable_io import atomic_write_bytes
from kernelforge.tracker import UsageAccumulator

log = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class DriverPreparation:
    """One preparation: the driver to author or repair, where it is audited, its deadline, evidence and attempts."""

    driver_path: str
    experiments_dir: str
    deadline_unix: float | None
    invocation_spec_file: str = ""
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


@dataclass(frozen=True)
class _AuthoringTurn:
    """One authoring session: the staging repository, the driver it edits, the evidence it must not touch, its log."""

    stage: Path
    stage_driver: Path
    evidence_paths: set[Path]
    prompt: str
    timeout_sec: int
    progress_log: list[str]


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

    @property
    def source_case_ms(self) -> dict[str, float]:
        return dict(self.reference.case_ms) if self.reference is not None else {}


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


def _restore_write_boundary(stage: Path, stage_driver: Path, evidence: dict[Path, tuple[bytes, int]]) -> list[str]:
    """Put the evidence back and remove stray outputs; what crossed the boundary, as sorted stage-relative paths."""
    allowed = {stage_driver.resolve(), *(path.resolve() for path in evidence)}
    unexpected = [
        path
        for path in stage.rglob("*")
        if ".git" not in path.parts and (path.is_file() or path.is_symlink()) and path.resolve() not in allowed
    ]
    evidence_changed = [
        path
        for path, (content, _mode) in evidence.items()
        if not path.is_file() or path.is_symlink() or path.parent.is_symlink() or path.read_bytes() != content
    ]
    _restore_evidence(evidence)
    for path in unexpected:
        try:
            path.unlink()
        except OSError:
            continue
    return sorted({path.relative_to(stage).as_posix() for path in [*unexpected, *evidence_changed]})


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
    turn: _AuthoringTurn,
    usage: UsageAccumulator | None = None,
) -> str:
    runtime = with_writable_sandbox(agent_runtime(config))
    backend = create_registered_backend(runtime)
    run_spec = AgentRunSpec(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=turn.prompt,
        role="flydsl driver preparation",
        cwd=str(turn.stage),
        writable=True,
        timeout_sec=turn.timeout_sec,
        target_files=[str(turn.stage_driver)],
        # Deliberately no driver_script, for the reason task_preparer records at its own AgentRunSpec: that field
        # declares the measurement surface the guard must defend, so it snapshots the driver as protected.
        protected_path_rules=FORGE_PROTECTED_PATHS,
        protected_globs=[path.name for path in turn.evidence_paths],
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
        progress_log=turn.progress_log,
    )
    result = await asyncio.wait_for(
        backend.run(run_spec, usage=usage),
        timeout=watchdog_timeout_sec(turn.timeout_sec),
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


@dataclass
class _DriverAuthoring:
    """The authoring turns of one preparation, the caller's driver they may replace, and what they have run into."""

    spec: RewriteSpec
    config: Config
    usage: UsageAccumulator | None
    preparation: DriverPreparation
    destination: Path
    original: bytes | None
    original_mode: int
    audit: Path
    last_preflight: DriverPreflight | None
    prior_failure: str = ""
    attempts_run: int = 0

    async def run(self, invocation_spec: bytes) -> DriverPreparationResult:
        """Stage the driver in an isolated repository and give each authoring turn its chance to make it conform."""
        deadline_unix = self.preparation.deadline_unix
        self.prior_failure = self.last_preflight.detail if self.last_preflight is not None else ""
        try:
            with tempfile.TemporaryDirectory(prefix="kernel_forge_rewrite_driver_") as temporary:
                stage = Path(temporary)
                stage_driver = stage / self.destination.name
                evidence = self._seed_stage(stage, stage_driver, invocation_spec)

                for attempt in range(1, self.preparation.max_attempts + 1):
                    remaining = _remaining(deadline_unix)
                    preflight_reserve = DEFAULT_REWRITE_BUDGET.driver_preflight_reserve_sec
                    if remaining <= preflight_reserve:
                        break
                    self.attempts_run = attempt
                    timeout_sec = max(1, int(min(float(self.config.agent_timeout_sec), remaining - preflight_reserve)))
                    prompt = _build_prompt(
                        spec=self.spec,
                        stage_driver=stage_driver,
                        invocation_spec_available=bool(invocation_spec),
                        prior_failure=self.prior_failure,
                    )
                    turn = _AuthoringTurn(stage, stage_driver, set(evidence), prompt, timeout_sec, [])
                    record = self.audit / f"attempt_{attempt:02d}"
                    await self._author(turn, record)
                    crossed = _restore_write_boundary(stage, stage_driver, evidence)
                    if crossed:
                        self.prior_failure = (
                            "the previous attempt violated the write boundary: "
                            + ", ".join(crossed)
                            + "; modify only the driver"
                        )
                        continue
                    if not stage_driver.is_file() or stage_driver.is_symlink():
                        self.prior_failure = "the previous attempt deleted the required driver"
                        continue

                    candidate_bytes = stage_driver.read_bytes()
                    try:
                        compile(candidate_bytes, str(self.destination), "exec", dont_inherit=True)
                    except (SyntaxError, ValueError) as error:
                        self.prior_failure = f"the generated driver is not valid Python: {error}"
                        continue

                    atomic_write_bytes(self.destination, candidate_bytes)
                    try:
                        self.last_preflight = preflight_rewrite_driver(
                            self.spec, str(self.destination), deadline_unix=deadline_unix
                        )
                    except Exception as error:  # isolation boundary: one attempt's deterministic driver preflight
                        log.warning(
                            "driver preparation attempt %d: preflight raised: %s", attempt, error, exc_info=True
                        )
                        _restore_driver(self.destination, self.original)
                        self.prior_failure = f"deterministic preflight raised {type(error).__name__}: {error}"
                        continue
                    _audit_driver(record, "driver_at_preflight.py", self.destination)
                    _audit_json(record, "preflight.json", _preflight_payload(self.last_preflight))
                    if self.last_preflight.ok:
                        return DriverPreparationResult(
                            ok=True,
                            attempts=attempt,
                            preflight=self.last_preflight,
                            audit_dir=str(self.audit),
                            wrote_driver=True,
                        )
                    _restore_driver(self.destination, self.original)
                    self.prior_failure = f"[{self.last_preflight.failure_class}] {self.last_preflight.detail}"
        except (OSError, RuntimeError, ValueError) as error:
            _restore_driver(self.destination, self.original)
            return self._failure(DRIVER_PREPARATION_FAILED, f"could not prepare the isolated driver workspace: {error}")

        _restore_driver(self.destination, self.original)
        deadline_reached = _remaining(deadline_unix) <= DEFAULT_REWRITE_BUDGET.driver_preflight_reserve_sec
        detail = self.prior_failure or "the preparation agent produced no conforming driver"
        if deadline_reached:
            return self._failure(
                DRIVER_PREPARATION_DEADLINE, f"driver preparation reached its deadline; last failure: {detail}"
            )
        return self._failure(DRIVER_PREPARATION_FAILED, detail)

    def _seed_stage(self, stage: Path, stage_driver: Path, invocation_spec: bytes) -> dict[Path, tuple[bytes, int]]:
        """Lay out the isolated repository: the driver to edit, the read-only evidence, and a git baseline."""
        if self.original is None:
            stage_driver.write_text('"""Rewrite measurement driver; prepared by KernelForge."""\n', encoding="utf-8")
        else:
            stage_driver.write_bytes(self.original)
            stage_driver.chmod(self.original_mode)
        evidence = _write_evidence(stage, self.spec, invocation_spec)
        _ensure_agent_git_workspace(stage)
        return evidence

    async def _author(self, turn: _AuthoringTurn, record: Path) -> None:
        """Run one authoring session, recording its prompt, the driver before and after it, and how the agent ended."""
        _audit_text(record, "prompt.md", turn.prompt)
        _audit_text(record, "system_prompt.md", _SYSTEM_PROMPT)
        _audit_driver(record, "driver_before.py", turn.stage_driver)
        try:
            output = await _run_agent(config=self.config, turn=turn, usage=self.usage)
            _audit_text(record, "agent_output.txt", output)
        except asyncio.TimeoutError:
            _audit_json(record, "agent_event.json", {"status": "timeout", "timeout_sec": turn.timeout_sec})
        except Exception as error:  # isolation boundary: one driver-authoring agent session
            log.warning(
                "driver preparation attempt %d: agent invocation failed: %s", self.attempts_run, error, exc_info=True
            )
            _audit_json(
                record, "agent_event.json", {"status": "error", "type": type(error).__name__, "error": str(error)}
            )
        finally:
            _audit_text(record, "agent_progress.txt", "\n".join(turn.progress_log))
            _audit_driver(record, "driver_after_agent.py", turn.stage_driver)

    def _failure(self, failure_class: str, error: str) -> DriverPreparationResult:
        return DriverPreparationResult(
            ok=False,
            attempts=self.attempts_run,
            preflight=self.last_preflight,
            failure_class=failure_class,
            error=error,
            audit_dir=str(self.audit),
        )


async def prepare_rewrite_driver(
    *,
    spec: RewriteSpec,
    config: Config,
    preparation: DriverPreparation,
    initial_preflight: DriverPreflight | None = None,
    usage: UsageAccumulator | None = None,
) -> DriverPreparationResult:
    """Author or repair one driver without exposing the caller's tree to writes."""

    destination = Path(preparation.driver_path).resolve()
    if destination in {Path(spec.source_kernel).resolve(), Path(spec.flydsl_kernel).resolve()}:
        return DriverPreparationResult(
            ok=False,
            failure_class=driver_contract.DRIVER_NOT_INDEPENDENT,
            error="the driver destination collides with a kernel under validation",
        )
    if "forge_experiments" in destination.parts:
        return DriverPreparationResult(
            ok=False,
            failure_class=driver_contract.DRIVER_NOT_INDEPENDENT,
            error="the driver destination is producer-owned experiment state and cannot own the correctness gate",
        )
    try:
        invocation_spec = _load_invocation_spec(preparation.invocation_spec_file)
    except (OSError, ValueError) as error:
        return DriverPreparationResult(ok=False, failure_class=INVOCATION_SPEC_INVALID, error=str(error))

    try:
        audit = _audit_root(preparation.experiments_dir, spec.operator_slug)
        original = destination.read_bytes() if destination.is_file() else None
        original_mode = destination.stat().st_mode & 0o777 if destination.is_file() else 0o644
    except OSError as error:
        return DriverPreparationResult(
            ok=False,
            failure_class=DRIVER_PREPARATION_FAILED,
            error=f"could not initialize rewrite driver preparation: {error}",
        )
    if initial_preflight is not None:
        _audit_json(audit, "initial_preflight.json", _preflight_payload(initial_preflight))
    authoring = _DriverAuthoring(
        spec, config, usage, preparation, destination, original, original_mode, audit, initial_preflight
    )
    return await authoring.run(invocation_spec)
