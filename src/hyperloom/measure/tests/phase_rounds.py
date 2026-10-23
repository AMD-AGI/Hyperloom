# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rounds a stand-in ``run_grid`` plays for the phase tests: they spend the seconds a real round spends and leave what the
launcher leaves beside each Magpie run's server log (``server_ready_at``, ``kv_metrics.json``, the engine's own log).

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

from hyperloom.orchestrator.actions.executors import VariantResult

T0 = 5_000.0
READY_ONLY_LOG = "INFO: Application startup complete.\n"
LM_EVAL_RESULTS = {"results": {"gsm8k": {"exact_match,flexible-extract": 0.64}}}
REPORT = {"success": True, "throughput": {"output_throughput": 1234.5, "completed_requests": 8, "num_prompts": 8}}

Script = Callable[[Path, dict[str, Any]], VariantResult]


class Clock(Protocol):
    now: float

    def advance(self, seconds: float) -> float: ...


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
    """One Magpie run: the seconds it spends and the marks it leaves in its directory."""

    def __init__(self, clock: Clock, directory: Path) -> None:
        self.clock, self.directory = clock, directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.marks: dict[str, float] = {}

    def boot(self, seconds: float, log: str = READY_ONLY_LOG) -> Run:
        self.clock.advance(seconds)
        (self.directory / "server.log").write_text(log, encoding="utf-8")
        (self.directory / "server_ready_at").write_text(f"{self.clock.now:.3f} {seconds:.3f}\n", encoding="utf-8")
        return self

    def found_ready(self) -> Run:
        (self.directory / "server_ready_at").write_text(f"{self.clock.now:.3f} 0.000\n", encoding="utf-8")
        return self

    def mark(self, phase: str, after: float) -> Run:
        self.marks[phase] = self.clock.advance(after)
        return self

    def end(self, after: float) -> Run:
        ended = self.clock.advance(after)
        phase_marks = [{"phase": phase, "mono": 1.0, "ts": at} for phase, at in self.marks.items()]
        kv = self.directory / "kv_metrics.json"
        kv.write_text(json.dumps({"phase_marks": phase_marks}), encoding="utf-8")
        os.utime(kv, (ended, ended))
        (self.directory / "benchmark_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
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
        """A graded InferenceX test: a warmup round that boots, warms up and evaluates, then a measured round."""
        return {"warmup": self._graded_warmup, "test": self._graded_measured}

    def agentx(self) -> dict[str, Script]:
        """An AgentX test: one call that runs a discarded warmup run, then the measured one."""
        return {"test": self._agentx}

    def failing_boot(self) -> dict[str, Script]:
        """A test whose server never got ready: nothing is stamped, the round fails."""
        return {"test": self._failing_boot}

    def _graded_warmup(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(7.779)
        run = Run(self.clock, slot).boot(82.221 * self.scale, log=self.vllm_log())
        run.mark("measured", 0.0).mark("eval", 60.0 * self.scale).end(50.0 * self.scale)
        self._spend(5.0)
        (slot / "eval_output").mkdir(exist_ok=True)
        (slot / "eval_output" / "results_2026-10-14T10-00-00.json").write_text(json.dumps(LM_EVAL_RESULTS))
        return result("warmup", run.directory)

    def _graded_measured(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(4.0)
        Run(self.clock, slot).found_ready().end(56.0 * self.scale)
        self._spend(40.0)
        return result("test", slot)

    def _agentx(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(10.0)
        Run(self.clock, slot / "warmup_round").boot(80.0 * self.scale, log=READY_ONLY_LOG).end(40.0 * self.scale)
        self._spend(30.0)
        measured = Run(self.clock, slot).found_ready()
        measured.mark("warmup", 15.0 * self.scale).mark("measured", 30.0 * self.scale)
        measured.mark("finished", 200.0 * self.scale).end(15.0 * self.scale)
        self._spend(30.0)
        workspace = slot / "benchmark_20261014_000000"  # the run's own directory, below the round's
        workspace.mkdir()
        (workspace / "benchmark_report.json").write_text(json.dumps(REPORT), encoding="utf-8")
        return result("test", workspace, nonfatal_warnings=["run_grid_warmup_discarded_first"])

    def _failing_boot(self, slot: Path, kwargs: dict[str, Any]) -> VariantResult:
        self._spend(1710.0)
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "server.log").write_text("[aiter] tuning gemm 17/4096\n", encoding="utf-8")
        return result("test", slot, status="failed", returncode=1, error="server died", error_class="x")


class StagedGrid:
    """A ``run_grid`` that plays the script of the variant's name."""

    def __init__(self, scripts: dict[str, Script]) -> None:
        self.scripts = scripts

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        name = kwargs["grid"][0].name
        return [self.scripts[name](Path(kwargs["output_root"]) / f"variant_00_{name}", kwargs)]
