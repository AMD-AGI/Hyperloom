# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measurement socket: one complete test, run by whichever backend is active (design 4.4.1).

A test is every measurement that shares one inference server: the server start, the optional warmup round, the
measured round(s), an accuracy round when the caller asked for one, and the teardown. ``run_test`` blocks; the caller
puts it in ``asyncio.to_thread``. When it returns, the inference server has been reclaimed.
"""

from __future__ import annotations

import importlib
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from hyperloom.contracts.measurement import AccuracyValue, AttachmentRef, FailureClass, PhaseTiming, SandboxFacts
from hyperloom.measure.jit_session import JitCacheSession
from hyperloom.measure.phases import PhaseRecorder

__all__ = [
    "EnvironmentRef",
    "MeasurementBackend",
    "MeasurementTimeout",
    "RoundOutcome",
    "RoundPlan",
    "SessionFacts",
    "TestCall",
    "TestMeta",
    "TestOutcome",
    "active_backend",
    "run_test",
    "set_active_backend",
]

Prep = Literal["grid", "conc_sweep"]
Kind = Literal["test", "profile", "accuracy", "repro"]
Phase = Literal["warmup", "measure", "accuracy"]


@dataclass(frozen=True)
class RoundPlan:
    """How many warmup rounds a test runs; the measured round count is one, as in ``run_grid``.

    ``warmup_rounds`` of ``None`` leaves the choice to ``run_grid``'s own default (its environment switch).
    """

    warmup_rounds: int | None = None
    measure_rounds: int = 1

    def __post_init__(self) -> None:
        if self.warmup_rounds not in (None, 0, 1):
            raise ValueError(
                f"warmup_rounds must be None, 0 or 1 (run_grid discards at most one), got {self.warmup_rounds!r}"
            )
        if self.measure_rounds != 1:
            raise ValueError(
                f"measure_rounds must be 1 (run_grid measures one round per variant), got {self.measure_rounds!r}"
            )

    @classmethod
    def from_env(cls) -> RoundPlan:
        """The plan ``run_grid`` applies when nobody states one."""
        return cls()


@dataclass(frozen=True)
class TestMeta:
    """Scheduling fields the test carries for its caller; the backend does not read them."""

    __test__ = False

    purpose: str | None = None
    priority: int | None = None
    branch_id: str | None = None
    parent_commit: str | None = None
    retest_of: str | None = None


@dataclass(frozen=True)
class EnvironmentRef:
    """Identifies the software stack a test is meant to run on; empty fields mean 'whatever is installed here'."""

    image_digest: str | None = None
    runtime_manifest_sha256: str | None = None

    @classmethod
    def current_stack(cls) -> EnvironmentRef:
        return cls()


@dataclass(frozen=True)
class SessionFacts:
    """Session-level facts the old flow keeps in its state and a runner takes from the instruction sheet."""

    reference_unset_envs: tuple[str, ...] = ()
    inferencex_path: str | None = None
    magpie_commit: str | None = None


@dataclass(frozen=True)
class TestCall:
    """One test: its measurement points, where its output goes, and how it is run."""

    __test__ = False

    points: tuple[Path, ...]
    output_dir: Path
    timeout_sec: int
    cwd: str
    magpie_python: str
    task_id: str
    kind: Kind = "test"
    prep: Prep = "grid"
    rounds: RoundPlan = field(default_factory=RoundPlan.from_env)
    accuracy: bool = False
    meta: TestMeta = field(default_factory=TestMeta)
    server_log_path: str | None = None
    result_dir: str | None = None
    silence_timeout_sec: float | None = None
    on_output: Callable[[], None] | None = None
    session_deadline_sec: float | None = None
    unset_envs: tuple[str, ...] = ()
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    environment: EnvironmentRef = field(default_factory=EnvironmentRef.current_stack)
    avoid_nodes: tuple[str, ...] = ()
    session_facts: SessionFacts | None = None
    serving_lease: Any = None
    #: What the request ships with it (a ``repro`` test's case). The sheet carries them; whoever runs the call has
    #: unpacked each under ``<cwd>/attach/<id>/`` (a file attachment is the file ``<cwd>/attach/<id>``).
    attachments: tuple[AttachmentRef, ...] = ()
    #: Where the backend stamps the phases of the test as it goes: a runner that has stamped its own passes its
    #: recorder, so what the backend adds follows it; a call with none gets a recorder of the backend's own.
    phases: PhaseRecorder | None = None
    #: The shared compiled-code cache this test uses (design 4.21): the backend seeds its directories before the server
    #: starts and stages what the compilers added once a test succeeds. ``None`` runs the test as it always did.
    jit: JitCacheSession | None = None


@dataclass(frozen=True)
class RoundOutcome:
    """One round of a test; ``point`` indexes ``TestCall.points``.

    ``accuracy`` is set on an ``accuracy`` round only: the scores the evaluation wrote, which is the test's one
    evaluation (in the warmup round of a graded test with a warmup, else in the measured round).
    ``nonfatal_warnings`` are the warnings ``run_grid`` wrote for the point, kept as written and carried by the first
    round of the point only, so a reader that joins the rounds' lists gets each warning once.
    """

    point: int
    phase: Phase
    output_dir: Path
    returncode: int
    accuracy: tuple[AccuracyValue, ...] = ()
    nonfatal_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestOutcome:
    """What a test produced; ``failure_class`` is empty on success.

    ``phases`` are the phases of the test the backend stamped, in order; a backend that stamps none leaves it empty.
    """

    __test__ = False

    returncode: int
    stdout: str
    stderr: str
    rounds: tuple[RoundOutcome, ...]
    sandbox: SandboxFacts
    request_id: str | None = None
    failure_class: FailureClass | Literal["infra", "unclear", "candidate"] | None = None
    failure_reason: str | None = None
    phases: tuple[PhaseTiming, ...] = ()


class MeasurementTimeout(subprocess.TimeoutExpired):
    """A test that ran out of time, with the phases it went through before it did."""

    def __init__(
        self, cmd: str, timeout: float, output: str | None = None, phases: tuple[PhaseTiming, ...] = ()
    ) -> None:
        super().__init__(cmd, timeout, output=output)
        self.phases = phases


class MeasurementBackend(Protocol):
    """What a measurement implementation provides."""

    name: str

    def run_test(self, call: TestCall) -> TestOutcome: ...


#: The backend used until another is chosen, by import path: the local backend imports this module for its types.
_DEFAULT_BACKEND = "hyperloom.measure.local:LocalMeasurementBackend"


class _Registry:
    """Holds the chosen backend; ``None`` means the default one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: MeasurementBackend | None = None

    def set(self, backend: MeasurementBackend | None) -> None:
        with self._lock:
            self._backend = backend

    def get(self) -> MeasurementBackend:
        with self._lock:
            if self._backend is not None:
                return self._backend
        module, _, attribute = _DEFAULT_BACKEND.partition(":")
        return getattr(importlib.import_module(module), attribute)()


_REGISTRY = _Registry()


def set_active_backend(backend: MeasurementBackend | None) -> None:
    """Choose the backend ``run_test`` uses; ``None`` returns to the local one."""
    _REGISTRY.set(backend)


def active_backend() -> MeasurementBackend:
    """The chosen backend, or the local one when none was chosen."""
    return _REGISTRY.get()


def run_test(call: TestCall) -> TestOutcome:
    """Run one complete test through the active measurement backend.

    A call with no point or no task id is refused here, and a backend that returns anything but a ``TestOutcome`` is a
    defect reported at the socket instead of surfacing later in a caller.
    """
    if not call.points or not call.task_id:
        raise ValueError("a test call needs at least one measurement point and a task id")
    outcome = _REGISTRY.get().run_test(call)
    if not isinstance(outcome, TestOutcome):
        raise TypeError(f"a measurement backend must return a TestOutcome, got {type(outcome).__name__}")
    return outcome
