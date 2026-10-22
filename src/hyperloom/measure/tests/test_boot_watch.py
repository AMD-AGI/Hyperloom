# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test whose server does not report ready within its boot limit ends there, as the candidate's failure.

The rounds run a real child under the launcher's own watchdog (``run_with_session_kill``), so the cancellation the
boot guard raises, the reaping and the ready stamp are the ones a sandbox sees; only ``run_grid`` is a stand-in.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.measure import boot_watch, local
from hyperloom.measure.boot_watch import BOOT_SHARE, boot_limit_sec, boot_seconds
from hyperloom.measure.local import LocalMeasurementBackend
from hyperloom.measure.returncodes import SERVER_BOOT_TIMEOUT_RC
from hyperloom.measure.socket import TestCall
from hyperloom.orchestrator.actions.executors import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    VariantResult,
    run_with_session_kill,
)
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS

#: A server that tunes kernels for as long as it is allowed to and never says it is ready.
TUNING_SERVER = (
    "import sys, time\n"
    "log = open(sys.argv[1], 'a', buffering=1)\n"
    "log.write('[aiter] TunableOp: tuning gemm 17/4096 (M=64, N=5120, K=5120)\\n')\n"
    "time.sleep(300)\n"
)
#: A server that is ready after a moment and answers for a while.
READY_SERVER = (
    "import sys, time\n"
    "log = open(sys.argv[1], 'a', buffering=1)\n"
    "time.sleep(0.3)\n"
    "log.write('INFO:     Application startup complete.\\n')\n"
    "time.sleep(2.5)\n"
)


def _point(directory: Path, framework: str = "sglang", **benchmark: Any) -> Path:
    path = directory / "point.yaml"
    body = {"framework": framework, "envs": {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}, **benchmark}
    path.write_text(yaml.safe_dump({"benchmark": body}), encoding="utf-8")
    return path


def _call(tmp_path: Path, point: Path, timeout_sec: int = 4) -> TestCall:
    return TestCall(
        points=(point,),
        output_dir=tmp_path / "out",
        timeout_sec=timeout_sec,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t-1",
    )


class _ServerGrid:
    """``run_grid`` with a real child process in place of Magpie and the server it starts."""

    def __init__(self, script: str) -> None:
        self.script = script
        self.rounds: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> list[VariantResult]:
        self.rounds.append(kwargs)
        name = kwargs["grid"][0].name
        slot = Path(kwargs["output_root"]) / f"variant_00_{name}"
        slot.mkdir(parents=True, exist_ok=True)
        log = slot / "server.log"
        proc = await asyncio.to_thread(
            run_with_session_kill, [sys.executable, "-c", self.script, str(log)], server_log_path=str(log), timeout=30
        )
        stopped = proc.returncode == ORCHESTRATOR_CANCELLED_RETURNCODE
        return [
            VariantResult(
                name=name,
                extra_server_args="",
                extra_envs={},
                status="skipped" if stopped else "failed" if proc.returncode else "succeeded",
                returncode=proc.returncode,
                workspace=str(slot),
                server_log_path=str(log),
                error_class=ORCHESTRATOR_CANCELLED_CLASS if stopped else "",
                error="the round was reaped" if stopped else None,
            )
        ]


@pytest.fixture(autouse=True)
def magpie_backend(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")


def _run(monkeypatch, tmp_path: Path, script: str, **point: Any):
    grid = _ServerGrid(script)
    monkeypatch.setattr(local, "run_grid", grid)
    started = time.monotonic()
    outcome = LocalMeasurementBackend().run_test(_call(tmp_path, _point(tmp_path, **point)))
    return outcome, time.monotonic() - started, grid


def test_a_server_that_never_gets_ready_ends_the_test_at_the_boot_limit_as_the_candidates_failure(
    monkeypatch, tmp_path
) -> None:
    outcome, elapsed, _ = _run(monkeypatch, tmp_path, TUNING_SERVER)

    assert outcome.returncode == SERVER_BOOT_TIMEOUT_RC
    assert outcome.failure_class == "candidate"
    assert outcome.failure_reason is not None
    assert outcome.failure_reason.startswith("server_boot_timeout: the server was not ready within 2s of the 4s")
    assert "TunableOp: tuning gemm 17/4096" in outcome.failure_reason, "the reason shows where the log stopped"
    assert [r.returncode for r in outcome.rounds] == [SERVER_BOOT_TIMEOUT_RC]
    assert 2 <= elapsed < 20, "the round ended at the limit, not at its 60 s cap"


def test_a_server_that_gets_ready_in_time_is_left_alone(monkeypatch, tmp_path) -> None:
    outcome, _, _ = _run(monkeypatch, tmp_path, READY_SERVER)

    assert (outcome.returncode, outcome.failure_class, outcome.failure_reason) == (0, None, None)


def test_a_stop_by_the_session_budget_is_not_taken_for_a_boot_timeout(monkeypatch, tmp_path) -> None:
    async def out_of_time(**kwargs: Any) -> list[VariantResult]:
        stopped = {"status": "skipped", "returncode": None, "error_class": "session_time_exhausted"}
        return [VariantResult(name=kwargs["grid"][0].name, extra_server_args="", extra_envs={}, **stopped)]

    monkeypatch.setattr(local, "run_grid", out_of_time)
    monkeypatch.setattr(boot_watch.BootGuard, "expired_after", property(lambda self: 2.0))

    outcome = LocalMeasurementBackend().run_test(_call(tmp_path, _point(tmp_path)))

    assert outcome.returncode == SESSION_TIME_EXHAUSTED_RETURNCODE
    assert (outcome.failure_class, outcome.failure_reason) == (None, "session_time_exhausted")


def _record_watches(monkeypatch) -> list[float]:
    watched: list[float] = []

    @contextlib.contextmanager
    def record(self, limit_sec: float) -> Iterator[None]:
        watched.append(limit_sec)
        yield

    monkeypatch.setattr(boot_watch.BootGuard, "watching", record)
    return watched


def test_only_a_round_that_boots_a_server_is_watched(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(local, "run_grid", _ServerGrid(READY_SERVER))
    watched = _record_watches(monkeypatch)

    async def boot_then_reuse(run: local._Run) -> None:
        run.add(0, await local._grid(run, 0, "boot"))
        run.add(0, await local._grid(run, 0, "reuse", server_already_ready=True))

    monkeypatch.setattr(local, "_runner_of", lambda call: boot_then_reuse)
    LocalMeasurementBackend().run_test(_call(tmp_path, _point(tmp_path)))

    assert watched == [2.0]


def test_the_limit_is_a_share_of_what_the_point_allows_and_never_of_more_than_the_round_may_take() -> None:
    assert boot_limit_sec({"framework": "sglang"}, 3600) == BOOT_SHARE * 3600 == 1800
    point_allows = {"framework": "vllm", "server_lifecycle": {"server_ready_timeout_s": 2700}}
    assert boot_limit_sec(point_allows, 3600) == BOOT_SHARE * 2700
    assert boot_limit_sec({**point_allows, "server_lifecycle": {"server_ready_timeout_s": 7200}}, 3600) == 1800


@pytest.mark.parametrize("framework", ["custom", "xdit", ""])
def test_a_round_whose_server_never_stamps_a_ready_banner_has_no_limit(framework: str) -> None:
    assert boot_limit_sec({"framework": framework}, 3600) is None


def test_the_literals_the_guard_copies_equal_their_definitions() -> None:
    kill = importlib.import_module("hyperloom.orchestrator.actions.executors._subprocess_kill")
    engine = importlib.import_module("hyperloom.orchestrator.actions.executors.bypass_engine")

    assert boot_watch._READY_STAMP == kill._READY_STAMP_NAME
    assert set(boot_watch._SERVER_FRAMEWORKS) == set(engine.SERVER_FRAMEWORKS)


def _stamp(directory: Path, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "server_ready_at").write_text(text, encoding="utf-8")


def test_the_boot_of_a_test_is_the_longest_boot_its_round_stamps_record(tmp_path: Path) -> None:
    _stamp(tmp_path / "variant_00_warmup", "1792596681.010 82.221\n")  # the stamps of a real graded test
    _stamp(tmp_path / "variant_00_test", "1792596811.700 0.000\n")  # its measured round found the server ready

    assert boot_seconds(tmp_path) == 82.221


def test_a_test_whose_rounds_all_found_a_server_ready_booted_none(tmp_path: Path) -> None:
    _stamp(tmp_path / "variant_00_test", "1792596811.700 0.000\n")

    assert boot_seconds(tmp_path) is None


def test_a_test_with_no_stamp_or_a_stamp_that_cannot_be_read_has_no_boot(tmp_path: Path) -> None:
    assert boot_seconds(tmp_path) is None
    assert boot_seconds(tmp_path / "never_landed") is None
    for text in ("", "1792596681.010\n", "1792596681.010 soon\n", "1792596681.010 nan\n", "1792596681.010 inf\n"):
        _stamp(tmp_path / "bad", text)
        assert boot_seconds(tmp_path) is None, text
