# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native builds that compile nothing, for the fake toolbox and the tests of the modules that read build state.

:class:`FakeBuildDriver` is a :class:`~hyperloom_multibranch.toolbox.pending_builds.BuildDriver` with an in-memory
store. :meth:`FakeBuildDriver.need` says which requests need a build of which component; requests that name the same
``source`` share a key, as equal stacks do. :meth:`FakeBuildDriver.script` decides how the next builds of a component
end: ready, a candidate compile error, or a machine failure. ``gate`` holds every build until it is set.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from hyperloom.contracts.measurement import TestRequest
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import FAILED, READY, BuildRecord
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native.service import Ensured

FAKE_IMAGE_DIGEST = "sha256:" + "0" * 64
FAKE_BUILDER_DIGEST = "sha256:" + "1" * 64
FAKE_GFX = "gfx942"
FAKE_RECIPE_VERSION = "fake/1"


class FakeBuildEnd(str, Enum):
    READY = "ready"
    COMPILE_ERROR = "compile_error"
    INFRA = "infra"


@dataclass(frozen=True)
class FakeBuild:
    """How one build ends, and the CPU and wall seconds it records."""

    end: FakeBuildEnd = FakeBuildEnd.READY
    detail: str = ""
    cpu_seconds: float = 120.0
    wall_seconds: float = 34.0


class FakeBuildDriver:
    """Builds in memory; ``eta`` is every build's estimate, ``ensured`` the keys asked for in order."""

    def __init__(self, *, eta: float = 34.0) -> None:
        self.eta = eta
        self.gate = asyncio.Event()
        self.gate.set()
        self.ensured: list[str] = []
        self._needs: list[tuple[Callable[[TestRequest], bool], str, str]] = []
        self._scripts: dict[str, list[FakeBuild]] = {}
        self._store: dict[str, BuildRecord] = {}

    def need(self, match: Callable[[TestRequest], bool], component: str, source: str = "") -> BuildInputs:
        """Requests ``match`` accepts need a build of ``component`` from ``source``; returns its inputs."""

        self._needs.append((match, component, source))
        return _inputs(component, source)

    def script(self, component: str, *builds: FakeBuild) -> None:
        """The next builds of ``component`` end as ``builds`` say, in order; then ready."""

        self._scripts.setdefault(component, []).extend(builds)

    def needs(self, request: TestRequest, environment: object) -> tuple[BuildInputs, ...]:
        return tuple(_inputs(component, source) for match, component, source in self._needs if match(request))

    def lookup(self, inputs: BuildInputs) -> BuildRecord | None:
        return self._store.get(inputs.key)

    def eta_seconds(self, inputs: BuildInputs) -> float:
        return self.eta

    async def ensure(self, inputs: BuildInputs) -> Ensured:
        await self.gate.wait()
        self.ensured.append(inputs.key)
        stored = self._store.get(inputs.key)
        if stored is not None:
            return Ensured(stored, reused=True)
        scripted = self._scripts.get(inputs.component)
        build = scripted.pop(0) if scripted else FakeBuild()
        if build.end is FakeBuildEnd.INFRA:
            raise BuildInfraError(build.detail or "the build runner was lost")
        record = self._store[inputs.key] = _record(inputs, build)
        return Ensured(record, reused=False)


def _inputs(component: str, source: str) -> BuildInputs:
    digest = hashlib.sha256(f"{component}/{source}".encode()).hexdigest()
    return BuildInputs(FAKE_IMAGE_DIGEST, FAKE_BUILDER_DIGEST, component, FAKE_GFX, digest, FAKE_RECIPE_VERSION)


def _record(inputs: BuildInputs, build: FakeBuild) -> BuildRecord:
    failed = build.end is FakeBuildEnd.COMPILE_ERROR
    return BuildRecord(
        key=inputs.key,
        inputs=inputs,
        status=FAILED if failed else READY,
        outputs=(),
        toolchain={},
        built_at="2026-01-01T00:00:00Z",
        build_seconds=build.cpu_seconds,
        wall_seconds=build.wall_seconds,
        failure=f"compile_error: {build.detail or 'error: expected expression'}" if failed else "",
    )


__all__ = ["FakeBuild", "FakeBuildDriver", "FakeBuildEnd"]
