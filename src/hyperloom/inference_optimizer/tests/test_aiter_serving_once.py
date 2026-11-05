# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch preflight reads only the rows aiter compiles for this card and leaves what it moves aside to the boot's
servers; the check of the image's own tables for this card builds what it moves aside once per container.

The image is the stand-in aiter of the measure tests with its GEMM module named as aiter's ``module_gemm_a8w8``, which is
generated from ``AITER_CONFIG_GEMM_A8W8``'s table. The table has a row tuned for this card (gfx950, 256 CUs) and one
tuned for another GPU (gfx942, 80 CUs); aiter compiles only the first into a module it builds for this card. The build is
the measure runner's prewarm, with a builder in place of the CPU build runner that writes what aiter would build.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.runner.native import prewarm
from hyperloom.measure.runner.native.tests import _aiter_standin as standin
from hyperloom.orchestrator.actions.executors import _aiter_jit
from hyperloom.orchestrator.actions.executors import _aiter_serving_once as once
from hyperloom.orchestrator.actions.executors.benchmark_run import launch as launch_mod

GFX, CU = "gfx950", 256
MODULE, VARIABLE = "module_gemm_a8w8", "AITER_CONFIG_GEMM_A8W8"
CARD_KERNEL = "a8w8_rowwise_256x64x128x128_32x32_1x2_card"
OTHER_GPU_KERNEL = "a8w8_rowwise_256x128x64x128_32x32_2x1_other_gpu"
TABLE = (
    "gfx,cu_num,M,N,K,kernelId,kernelName\n"
    f"{GFX},{CU},64,57344,8192,46,{CARD_KERNEL}\n"
    f"gfx942,80,64,57344,8192,45,{OTHER_GPU_KERNEL}\n"
)


def module_of(*kernels: str) -> bytes:
    return b"\x7fELF " + " ".join(kernels).encode()


@pytest.fixture
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The image's aiter on a gfx950 card of 256 CUs; its ``module_gemm_a8w8`` was built from an older table."""
    return serving_image(tmp_path, monkeypatch)


def serving_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The aiter of :func:`image`, made below ``tmp_path`` where the preflight finds it, on a gfx950 card of 256 CUs;
    this host's lock directory is below ``tmp_path`` too."""
    package = standin.aiter_image(tmp_path / "image")
    config_file = package / "jit" / "optCompilerConfig.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config[MODULE] = config.pop(standin.MODULE)
    config_file.write_text(json.dumps(config), encoding="utf-8")
    (package / "jit" / f"{standin.MODULE}.so").unlink()
    (package / "jit" / f"{MODULE}.so").write_bytes(module_of("a8w8_rowwise_retired"))
    (package / "configs" / "a8w8_tuned_gemm.csv").write_text(TABLE, encoding="utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(package / "jit"))
    for name in ("GPU_ARCHS", "CU_NUM", "AITER_JIT_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_aiter_jit, "card_target", lambda: (GFX, CU))
    monkeypatch.setattr(once, "HOST_DIR", tmp_path / "host")
    return package


class Prewarm:
    """The prewarm the measure runner gives the preflight as its build, with a builder that writes ``kernels`` into each module it is asked for (aiter
    writes the kernels of the card's rows), or raises ``failure``; ``wait`` holds the build that long."""

    def __init__(self, *kernels: str, failure: Exception | None = None, wait: float = 0.0) -> None:
        self.kernels = kernels or (CARD_KERNEL,)
        self.failure, self.wait = failure, wait
        self.calls: list[tuple[tuple[str, ...], str, int]] = []

    def __call__(self, package: Path, variables: Sequence[str], gfx: str, cu_num: int) -> dict[str, Any]:
        self.calls.append((tuple(variables), gfx, cu_num))
        time.sleep(self.wait)
        return asdict(prewarm.prewarm(package, self._build, gfx=gfx, cu_num=cu_num, tables=variables))

    def _build(self, package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        if self.failure is not None:
            raise self.failure
        built = {name: scratch / f"{name}.so" for name in modules}
        for path in built.values():
            path.write_bytes(module_of(*self.kernels))
        return built


def covers_the_card(image: Path) -> bool:
    table = image / "configs" / "a8w8_tuned_gemm.csv"
    return _aiter_jit.serving_modules_cover_rows(
        image / "jit", (MODULE,), _aiter_jit.csv_jit_kernel_rows(table, frozenset({(GFX, CU)}))
    )


def test_only_the_rows_aiter_compiles_for_this_card_decide_whether_a_module_is_moved_aside(image, monkeypatch) -> None:
    """The image's modules lacked only kernels of rows tuned for other GPUs, and were moved aside at every session."""
    (image / "jit" / f"{MODULE}.so").write_bytes(module_of(CARD_KERNEL))

    covered = _aiter_jit.prepare_serving_so_for_csvs({})

    assert covered == {"action": "skip", "jit_dir": str(image / "jit"), "targets": ["gfx950/256"]}
    monkeypatch.setattr(_aiter_jit, "card_target", lambda: None)
    assert _aiter_jit.prepare_serving_so_for_csvs({})["action"] == "invalidate", "every row counts on an unknown card"


def test_a_module_moved_aside_is_built_once_installed_and_kept_by_the_next_preflight(image, monkeypatch) -> None:
    runner = Prewarm()

    first = once.build_image_own_once(runner)

    assert first.outcome["action"] == "invalidate" and first.outcome["variables"] == [VARIABLE]
    assert first.rebuild == once.Rebuild(built=(MODULE,), seconds=first.rebuild.seconds)
    assert runner.calls == [((VARIABLE,), GFX, CU)]
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of(CARD_KERNEL)
    assert covers_the_card(image), "the preflight's own test passes on what was built"
    entry = _aiter_jit.read_prewarm_manifest(image / "jit")[MODULE]
    assert entry["gfx"] == GFX and entry["cu_num"] == CU
    assert "built module_gemm_a8w8 once from the image's own tables" in first.summary()

    again = once.build_image_own_once(runner)
    assert again.outcome["action"] == "skip" and again.rebuild is None and len(runner.calls) == 1

    (image / "configs" / "a8w8_tuned_gemm.csv").write_text(TABLE + f"{GFX},{CU},1,2,3,4,{OTHER_GPU_KERNEL}\n", "utf-8")
    newer = once.build_image_own_once(runner)
    assert newer.outcome["action"] == "invalidate", "built from other tables than the image has now"


def test_a_prewarmed_module_is_kept_only_for_the_card_it_was_built_for(image, monkeypatch) -> None:
    """With every row counting (the card unknown), the module built for gfx950/256 lacks the gfx942 row's kernel: it
    is not the module the servers would build, and is moved aside."""
    runner = Prewarm()
    once.build_image_own_once(runner)
    assert once.build_image_own_once(runner).outcome["action"] == "skip"

    monkeypatch.setattr(_aiter_jit, "card_target", lambda: None)
    unknown = once.prepare_serving_so_once({}, {})

    assert unknown.outcome["action"] == "invalidate" and unknown.outcome["removed"] == [
        str(image / "jit" / f"{MODULE}.so")
    ]
    assert unknown.rebuild == once.Rebuild(left=(MODULE,), why="the launch leaves them to its servers")


def test_two_sessions_at_once_build_it_once(image, monkeypatch) -> None:
    runner = Prewarm(wait=0.5)
    start = threading.Barrier(2)
    results: list[once.ServingPreflight] = []

    def session() -> None:
        start.wait()
        results.append(once.build_image_own_once(runner))

    threads = [threading.Thread(target=session) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(runner.calls) == 1
    assert sorted(result.outcome["action"] for result in results) == ["invalidate", "skip"]
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of(CARD_KERNEL)


def test_a_failed_build_leaves_the_module_to_the_servers_and_is_not_tried_again(image, monkeypatch) -> None:
    runner = Prewarm(failure=prewarm.PrewarmError("the build failed: compile_error"))

    failed = once.build_image_own_once(runner)

    assert failed.outcome["action"] == "invalidate" and failed.rebuild.built == ()
    assert "compile_error" in failed.rebuild.why
    assert not (image / "jit" / f"{MODULE}.so").exists(), "moved aside, as before: the servers build it"
    assert "not built once" in failed.summary() and "each server builds them itself" in failed.summary()

    (image / "jit" / f"{MODULE}.so").write_bytes(module_of("a8w8_rowwise_retired"))
    again = once.build_image_own_once(runner)

    assert again.outcome["action"] == "invalidate" and len(runner.calls) == 1
    assert again.rebuild.why.startswith("tried once on this container already")


class _Crashing(Prewarm):
    """A build that raises what the prewarm does not expect: a defect in it, not a failed build."""

    def _build(self, package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        raise RuntimeError("a defect of the build")


def test_a_build_that_raises_is_not_tried_again_by_every_later_test(image) -> None:
    """The build runs in the caller's process: what it raises reaches the caller, once, and the attempt is on record."""
    runner = _Crashing()
    with pytest.raises(RuntimeError, match="a defect of the build"):
        once.build_image_own_once(runner)
    (image / "jit" / f"{MODULE}.so").write_bytes(module_of("a8w8_rowwise_retired"))

    again = once.build_image_own_once(runner)

    assert again.outcome["action"] == "invalidate" and len(runner.calls) == 1
    assert again.rebuild.why.startswith("tried once on this container already")
    assert again.rebuild.why.endswith("started, and did not end")


def test_a_build_that_does_not_cover_the_cards_rows_is_not_installed(image, monkeypatch) -> None:
    result = once.build_image_own_once(Prewarm(OTHER_GPU_KERNEL))

    assert result.rebuild.built == () and "lack kernels of gfx950/256 rows of a8w8_tuned_gemm.csv" in result.rebuild.why
    assert not (image / "jit" / f"{MODULE}.so").exists()
    assert _aiter_jit.read_prewarm_manifest(image / "jit") == {}


def test_an_unknown_card_moves_nothing_for_a_build_once(image, monkeypatch) -> None:
    """Every row would count, and the image's module lacks the other GPU's: moved aside, it would be built by no one,
    as no test's server builds into the image."""
    runner = Prewarm()
    (image / "jit" / f"{MODULE}.so").write_bytes(module_of(CARD_KERNEL))
    monkeypatch.setattr(_aiter_jit, "card_target", lambda: None)

    result = once.build_image_own_once(runner)

    assert result.outcome == {"action": "noop", "reason": "this card's target is unknown"} and runner.calls == []
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of(CARD_KERNEL)


@pytest.mark.asyncio
async def test_the_launch_logs_the_decision_at_the_level_optimize_logs_at(image, monkeypatch, caplog) -> None:
    """``hyperloom optimize`` logs at WARNING without ``-v``: an INFO line would not reach its log."""
    monkeypatch.setattr(launch_mod, "_COVERED_SAID", set())
    caplog.set_level(logging.WARNING)

    await launch_mod._prepare_aiter_serving_so({})
    await launch_mod._prepare_aiter_serving_so({})
    await launch_mod._prepare_aiter_serving_so({})

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == launch_mod.log.name
    ]
    assert len(lines) == 2, "a move every time, nothing moved once"
    moved, covered = lines
    assert "outran the serving .so for gfx950/256; moved 1 module(s) aside" in moved
    assert f"not built once (the launch leaves them to its servers); left for the servers to build: {MODULE}" in moved
    assert covered.endswith("the serving .so covers the CSVs this boot loads for gfx950/256; nothing moved")


@pytest.mark.asyncio
async def test_the_launch_moves_a_module_aside_and_starts_no_build(image, monkeypatch) -> None:
    """The launch of a boot leaves what it moved aside to its servers: it runs no prewarm, in no process of its own."""
    started: list[Any] = []

    def no_process(*args: Any, **kwargs: Any) -> None:
        started.append(args[0] if args else kwargs.get("args"))
        raise OSError("this test starts no process")

    monkeypatch.setattr(subprocess, "Popen", no_process)

    await launch_mod._prepare_aiter_serving_so({})

    assert started == [], "the launch started a process"
    assert not (image / "jit" / f"{MODULE}.so").exists(), "moved aside: the boot's servers build it"
    assert _aiter_jit.read_prewarm_manifest(image / "jit") == {}


def test_serving_modules_outside_the_images_aiter_jit_are_not_built_into_it(image, monkeypatch, tmp_path) -> None:
    elsewhere = tmp_path / "elsewhere" / "jit"
    elsewhere.mkdir(parents=True)
    shutil.copyfile(image / "jit" / f"{MODULE}.so", elsewhere / f"{MODULE}.so")
    monkeypatch.setattr(_aiter_jit, "resolve_serving_context", lambda *args, **kwargs: (image, elsewhere / "build"))
    runner = Prewarm()

    result = once.build_image_own_once(runner)

    assert result.outcome["removed"] == [str(elsewhere / f"{MODULE}.so")] and runner.calls == []
    assert result.rebuild.why == f"the serving modules are not the image's own aiter/jit: {elsewhere}"
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of("a8w8_rowwise_retired")


def test_a_host_directory_that_cannot_be_used_leaves_the_preflight_to_run_unlocked(
    image, monkeypatch, tmp_path, caplog
) -> None:
    """The move protects the boot from a registry mismatch: a lock that cannot be had is no reason to skip it. Without
    the lock nothing is built once: another session's preflight could move the modules under the build."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the directory would be", encoding="utf-8")
    monkeypatch.setattr(once, "HOST_DIR", blocker / "host")
    runner = Prewarm()
    caplog.set_level(logging.WARNING)

    result = once.build_image_own_once(runner)

    assert not (image / "jit" / f"{MODULE}.so").exists(), "moved aside: the boot builds it"
    assert runner.calls == [] and result.rebuild == once.Rebuild(left=(MODULE,), why="this host's lock cannot be used")
    lines = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any(
        line.startswith(
            f"aiter serving preflight: this host's lock {blocker / 'host' / 'preflight.lock'} cannot be used"
        )
        and line.endswith("the prewarm of the image's own serving modules runs unlocked")
        for line in lines
    )


def forged_attempt(image: Path) -> list[dict[str, Any]]:
    """A record of attempts that says this container tried to build the image's module for this card already."""
    return [
        {
            "jit_dir": str(image / "jit"),
            "target": f"{GFX}/{CU}",
            "tables_sha256": _aiter_jit.tuned_tables_sha256(image / "configs"),
            "variables": [VARIABLE],
            "result": "forged",
            "at": "2026-01-01T00:00:00Z",
        }
    ]


def unlocked_warnings(caplog: pytest.LogCaptureFixture, doing: str) -> list[str]:
    lock = once.HOST_DIR / "preflight.lock"
    return [
        line
        for line in (record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)
        if line.startswith(f"aiter serving preflight: this host's lock {lock} cannot be used")
        and line.endswith(f"{doing} runs unlocked")
    ]


def test_a_lock_planted_as_a_link_is_not_followed(image, tmp_path, caplog) -> None:
    """A link at the lock's name would have every boot empty the file it points at: the lock is never opened through a
    link, and the preflight runs unlocked and says so."""
    victim = tmp_path / "victim"
    victim.write_text("a file the lock points at", encoding="utf-8")
    once.HOST_DIR.mkdir(mode=0o700)
    (once.HOST_DIR / "preflight.lock").symlink_to(victim)
    caplog.set_level(logging.WARNING)

    once.prepare_serving_so_once({}, {})

    assert victim.read_text(encoding="utf-8") == "a file the lock points at"
    assert unlocked_warnings(caplog, "the launch preflight of a boot")
    assert not (image / "jit" / f"{MODULE}.so").exists(), "moved aside all the same: the boot builds it"


@pytest.mark.parametrize("planted", ["group-writable", "others-writable", "another-account"])
def test_a_host_directory_this_account_alone_cannot_write_is_neither_locked_in_nor_read(
    image, monkeypatch, caplog, planted
) -> None:
    """Every account on the host shares ``/tmp``: a directory under the lock's name that another account made, or can
    write, could hold a lock that is a link and a record that says the build was tried. The preflight takes no lock
    there and reads and writes no record: it runs unlocked and says so, and nothing is built once."""
    host = once.HOST_DIR
    host.mkdir(mode=0o700)
    (host / "attempts.json").write_text(json.dumps(forged_attempt(image)), encoding="utf-8")
    if planted == "another-account":
        uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    else:
        host.chmod(0o770 if planted == "group-writable" else 0o707)
    before = {path.name: path.read_bytes() for path in host.iterdir()}
    runner = Prewarm()
    caplog.set_level(logging.WARNING)

    result = once.build_image_own_once(runner)

    assert runner.calls == [] and result.rebuild == once.Rebuild(left=(MODULE,), why="this host's lock cannot be used")
    assert {path.name: path.read_bytes() for path in host.iterdir()} == before, "nothing is written in it"
    assert unlocked_warnings(caplog, "the prewarm of the image's own serving modules")


def test_a_record_of_attempts_planted_as_a_link_is_not_read(image, tmp_path) -> None:
    host = once.HOST_DIR
    host.mkdir(mode=0o700)
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(forged_attempt(image)), encoding="utf-8")
    (host / "attempts.json").symlink_to(forged)
    runner = Prewarm()

    result = once.build_image_own_once(runner)

    assert len(runner.calls) == 1 and result.rebuild.built == (MODULE,), "the forged record did not stop the build"
    assert not (host / "attempts.json").is_symlink() and json.loads(forged.read_text(encoding="utf-8"))[0][
        "result"
    ] == ("forged")


def test_a_preflight_that_waits_for_the_lock_says_who_holds_it_and_what_it_does(image, monkeypatch, caplog) -> None:
    host = once.HOST_DIR
    host.mkdir(mode=0o700, parents=True)
    holder = {"pid": 4242, "since": "2026-11-03T00:00:00Z", "doing": "building module_gemm_a8w8 for gfx950/256"}
    (host / "preflight.lock").write_text(json.dumps(holder), encoding="utf-8")
    flock = fcntl.flock

    def held_by_another(descriptor: int, operation: int) -> None:
        if operation & fcntl.LOCK_NB:
            raise BlockingIOError(11, "Resource temporarily unavailable")
        flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", held_by_another)
    caplog.set_level(logging.WARNING)

    once.prepare_serving_so_once({}, {})

    lines = [record.getMessage() for record in caplog.records if record.name == once.log.name]
    assert lines[0] == (
        f"aiter serving preflight: the launch preflight of a boot waits for this host's lock {host / 'preflight.lock'}, "
        "held by pid 4242 since 2026-11-03T00:00:00Z: building module_gemm_a8w8 for gfx950/256"
    )
    assert lines[1].startswith("aiter serving preflight: the launch preflight of a boot got this host's lock after")
    assert (host / "preflight.lock").read_text(encoding="utf-8") == "", "the holder's word goes with its lock"


class _Interrupted(BaseException):
    """What stops a wait that is no error of the lock, such as a signal."""


def test_a_wait_for_the_lock_that_is_interrupted_lets_its_descriptor_go(image, monkeypatch) -> None:
    waited: list[int] = []

    def interrupted(descriptor: int, operation: int) -> None:
        if operation & fcntl.LOCK_NB:
            raise BlockingIOError
        waited.append(descriptor)
        raise _Interrupted

    monkeypatch.setattr(fcntl, "flock", interrupted)

    with pytest.raises(_Interrupted):
        once.prepare_serving_so_once({}, {})

    with pytest.raises(OSError):
        os.fstat(waited[0])


class _Watched(Prewarm):
    """A prewarm that notes, when its build starts, the warnings given so far and what the lock's holder says."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        super().__init__()
        self.caplog = caplog
        self.said: list[str] = []
        self.holder: dict[str, Any] = {}

    def _build(self, package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        self.said = [record.getMessage() for record in self.caplog.records if record.levelno == logging.WARNING]
        self.holder = json.loads((once.HOST_DIR / "preflight.lock").read_text(encoding="utf-8"))
        return super()._build(package, modules, scratch)


def test_a_build_says_what_it_builds_for_which_card_before_it_starts(image, monkeypatch, caplog) -> None:
    runner = _Watched(caplog)
    caplog.set_level(logging.WARNING)

    once.build_image_own_once(runner)

    assert (
        f"aiter serving preflight: building {MODULE} once for {GFX}/{CU} before this boot (the prewarm of {VARIABLE})"
    ) in runner.said
    assert f"aiter prewarm: building {MODULE} for {GFX}/{CU} on CPU, about 65 s" in " ".join(runner.said), (
        "the stand-in's module is neither a CK-tile one nor generated by aiter's gen_instances: 65 s"
    )
    assert runner.holder["pid"] == os.getpid()
    assert runner.holder["doing"] == f"building {MODULE} for {GFX}/{CU} (the prewarm)"


def test_a_module_aiter_cannot_build_by_name_is_reported_as_left_to_the_servers(image, monkeypatch) -> None:
    config_file = image / "jit" / "optCompilerConfig.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config[MODULE]["third_party"] = ["'ck'"]
    config_file.write_text(json.dumps(config), encoding="utf-8")

    result = once.build_image_own_once(Prewarm())

    assert result.rebuild == once.Rebuild(
        left=(MODULE,), seconds=result.rebuild.seconds, why="aiter cannot build them by name"
    )
    assert result.summary().endswith(
        f"; not built once (aiter cannot build them by name); left for the servers to build: {MODULE} "
        "(each server builds them itself)"
    )


def test_a_build_that_failed_for_one_card_target_is_still_tried_for_another(image, monkeypatch) -> None:
    """The record keys an attempt by the target too: a card partitioned to another CU count builds other rows."""
    table = image / "configs" / "a8w8_tuned_gemm.csv"
    table.write_text(TABLE + f"{GFX},128,64,57344,8192,47,a8w8_rowwise_partitioned\n", encoding="utf-8")
    runner = Prewarm(failure=prewarm.PrewarmError("the build failed: compile_error"))
    once.build_image_own_once(runner)
    (image / "jit" / f"{MODULE}.so").write_bytes(module_of("a8w8_rowwise_retired"))

    monkeypatch.setattr(_aiter_jit, "card_target", lambda: (GFX, 128))
    again = once.build_image_own_once(runner)

    assert again.outcome["targets"] == [f"{GFX}/128"]
    assert [call[1:] for call in runner.calls] == [(GFX, CU), (GFX, 128)]
