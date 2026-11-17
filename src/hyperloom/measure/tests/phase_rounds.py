# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rounds a stand-in ``run_grid`` plays for the phase tests: they spend made-up seconds the way a real round does and
leave what an inference_suite round leaves in its run_dir (``server_ready_at``, ``kv_metrics.json``, the engine's own
log, ``report.json``), in the run_dir and the variant's slot ``run_grid`` and the round make, by the helpers they make
them with.

The same rounds serve a test that owns the clock, so every length can be asserted exactly, and the runner of a hosted
test, which spends real seconds scaled down: ``Rounds(RealClock(), 0.002)`` plays the 82 s boot in 0.16 s.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from hyperloom.measure.tests.run_dirs import write_run_dir
from hyperloom.orchestrator.actions.executors import VariantResult, variant_slot
from hyperloom.orchestrator.actions.executors._suite.rounds import _next_run_dir
from inference_suite.contracts import ReadyStamp, RunLayout

T0 = 5_000.0
READY_ONLY_LOG = "INFO: Application startup complete.\n"
SCORES = {"exact_match,flexible-extract": 0.64}
FIXED = {"output_throughput": 1234.5, "completed_requests": 8, "requested_requests": 8}

Script = Callable[[Path, dict[str, Any]], VariantResult]


class Clock(Protocol):
    now: float

    def advance(self, seconds: float) -> float:
        """Let ``seconds`` pass and return the time after them."""


class FakeClock:
    """The seconds a test spends; only the test moves it."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class RealClock:
    """The seconds really pass."""

    @property
    def now(self) -> float:
        return time.time()

    def advance(self, seconds: float) -> float:
        time.sleep(seconds)
        return time.time()


def result(name: str, slot: Path, **fields: Any) -> VariantResult:
    defaults: dict[str, Any] = {"extra_server_args": "", "extra_envs": {}, "status": "succeeded", "returncode": 0}
    return VariantResult(name=name, workspace=str(slot), **{**defaults, **fields})


class Run:
    """One round's run: the seconds it spends and the marks it leaves in the next run_dir of the variant's ``slot``."""

    def __init__(self, clock: Clock, slot: Path) -> None:
        self.clock, self.directory = clock, _next_run_dir(slot)
        self.directory.mkdir(parents=True)
        self.layout = RunLayout(self.directory)
        self.marks: dict[str, float] = {}

    def boot(self, seconds: float, log: str = READY_ONLY_LOG) -> Run:
        self.clock.advance(seconds)
        self.layout.server_log.write_text(log, encoding="utf-8")
        self.layout.write_ready_stamp(ReadyStamp(self.clock.now, seconds))
        return self

    def found_ready(self) -> Run:
        self.layout.write_ready_stamp(ReadyStamp(self.clock.now, 0.0))
        return self

    def mark(self, phase: str, after: float) -> Run:
        self.marks[phase] = self.clock.advance(after)
        return self

    def end(self, after: float, scores: dict[str, float] | None = None) -> Run:
        ended = self.clock.advance(after)
        phase_marks = [{"phase": phase, "mono": 1.0, "ts": at} for phase, at in self.marks.items()]
        kv = self.layout.kv_metrics
        kv.write_text(json.dumps({"phase_marks": phase_marks}), encoding="utf-8")
        os.utime(kv, (ended, ended))
        write_run_dir(self.directory, FIXED, scores, eval_code=None if scores is None else 0)
        return self


class Rounds:
    """The rounds of the tests a session runs, each as the ``run_grid`` call that plays it; lengths are in seconds times
    ``scale``."""

    def __init__(self, clock: Clock, scale: float = 1.0) -> None:
        self.clock, self.scale = clock, scale

    def _spend(self, seconds: float) -> None:
        self.clock.advance(seconds * self.scale)

    def vllm_log(self) -> str:
        """What a vLLM server says of its boot: the weights took 40 s, the compile 10 s, the graphs 12 s."""
        return (
            f"Loading weights took {40.0 * self.scale:.4f} seconds\ntorch.compile takes {10.0 * self.scale:.4f} s in total\n"
            f"Graph capturing finished in {12.0 * self.scale:.4f} secs, took 0.5 GiB\n{READY_ONLY_LOG}"
        )

    def graded(self) -> dict[str, Script]:
        """A graded InferenceX test: a warmup round that boots and warms up, a measured round, then a round that only
        evaluates, all on the server the warmup booted."""
        return {"warmup": self._warmup, "test": self._measured, "eval": self._evaluation}

    def agentx(self) -> dict[str, Script]:
        """An AgentX test: one measured round that boots, and whose client has phases of its own."""
        return {"test": self._agentx_measured}

    def failing_boot(self) -> dict[str, Script]:
        """A test whose server never got ready: nothing is stamped, the round that was to boot it fails."""
        return {"warmup": self._failing_boot, "test": self._failing_boot}

    def _warmup(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(8.0)
        run = Run(self.clock, slot).boot(82.0 * self.scale, log=self.vllm_log()).end(60.0 * self.scale)
        self._spend(5.0)
        return result("warmup", run.directory)

    def _measured(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(4.0)
        run = Run(self.clock, slot).found_ready().end(56.0 * self.scale)
        self._spend(40.0)
        return result("test", run.directory)

    def _evaluation(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(3.0)
        run = Run(self.clock, slot).found_ready().mark("eval", 2.0 * self.scale).end(50.0 * self.scale, SCORES)
        self._spend(2.0)
        return result("eval", run.directory, note="evaluation_only")

    def _agentx_measured(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(5.0)
        measured = Run(self.clock, slot).boot(80.0 * self.scale, log=READY_ONLY_LOG)
        measured.mark("warmup", 15.0 * self.scale).mark("measured", 30.0 * self.scale)
        measured.mark("finished", 200.0 * self.scale).end(15.0 * self.scale)
        self._spend(30.0)
        return result("test", measured.directory)

    def _failing_boot(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(1000.0)
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "server.log").write_text("[aiter] tuning gemm 17/4096\n", encoding="utf-8")
        name = kwargs["grid"][0].name
        return result(name, slot, status="failed", returncode=1, error="server died", error_class="x")


class StagedGrid:
    """A ``run_grid`` that plays the script of the variant's name."""

    def __init__(self, scripts: dict[str, Script]) -> None:
        self.scripts = scripts

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        name = kwargs["grid"][0].name
        return [self.scripts[name](variant_slot(Path(kwargs["output_root"]), 0, name), kwargs)]
