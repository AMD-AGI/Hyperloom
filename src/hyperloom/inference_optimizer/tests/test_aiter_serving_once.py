# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch preflight reads only the rows aiter compiles for this card, and builds what it moves aside once per
container, from the image's own tables (board #169).

The image is the stand-in aiter of the measure tests with its GEMM module named as aiter's ``module_gemm_a8w8``, which is
generated from ``AITER_CONFIG_GEMM_A8W8``'s table. The table has a row tuned for this card (gfx950, 256 CUs) and one
tuned for another GPU (gfx942, 80 CUs); aiter compiles only the first into a module it builds for this card. The prewarm
runs as the preflight runs it, with a builder in place of the CPU build runner that writes what aiter would build.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.jit.prewarmed import read_manifest
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
    """The prewarm the preflight runs, with a builder that writes ``kernels`` into each module it is asked for (aiter
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
    return _aiter_jit.serving_modules_cover_csv(image / "jit", (MODULE,), table, frozenset({(GFX, CU)}))


def test_only_the_rows_aiter_compiles_for_this_card_decide_whether_a_module_is_moved_aside(image, monkeypatch) -> None:
    """#169: the image's modules lacked only kernels of rows tuned for other GPUs, and were moved aside at every session."""
    (image / "jit" / f"{MODULE}.so").write_bytes(module_of(CARD_KERNEL))

    covered = _aiter_jit.prepare_serving_so_for_csvs({})

    assert covered == {"action": "skip", "jit_dir": str(image / "jit"), "targets": ["gfx950/256"]}
    monkeypatch.setattr(_aiter_jit, "card_target", lambda: None)
    assert _aiter_jit.prepare_serving_so_for_csvs({})["action"] == "invalidate", "every row counts on an unknown card"


def test_a_module_moved_aside_is_built_once_installed_and_kept_by_the_next_preflight(image, monkeypatch) -> None:
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)

    first = once.prepare_serving_so_once({}, {})

    assert first.outcome["action"] == "invalidate" and first.outcome["variables"] == [VARIABLE]
    assert first.rebuild == once.Rebuild(built=(MODULE,), seconds=first.rebuild.seconds)
    assert runner.calls == [((VARIABLE,), GFX, CU)]
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of(CARD_KERNEL)
    assert covers_the_card(image), "the preflight's own test passes on what was built"
    entry = read_manifest(image / "jit")[MODULE]
    assert entry["gfx"] == GFX and entry["cu_num"] == CU
    assert "built module_gemm_a8w8 once from the image's own tables" in first.summary()

    again = once.prepare_serving_so_once({}, {})
    assert again.outcome["action"] == "skip" and again.rebuild is None and len(runner.calls) == 1

    (image / "configs" / "a8w8_tuned_gemm.csv").write_text(TABLE + f"{GFX},{CU},1,2,3,4,{OTHER_GPU_KERNEL}\n", "utf-8")
    newer = once.prepare_serving_so_once({}, {})
    assert newer.outcome["action"] == "invalidate", "built from other tables than the image has now"


def test_a_prewarmed_module_is_kept_only_for_the_card_it_was_built_for(image, monkeypatch) -> None:
    """With every row counting (the card unknown), the module built for gfx950/256 lacks the gfx942 row's kernel: it
    is not the module the servers would build, and is moved aside."""
    monkeypatch.setattr(once, "_run_prewarm", Prewarm())
    once.prepare_serving_so_once({}, {})
    assert once.prepare_serving_so_once({}, {}).outcome["action"] == "skip"

    monkeypatch.setattr(_aiter_jit, "card_target", lambda: None)
    unknown = once.prepare_serving_so_once({}, {})

    assert unknown.outcome["action"] == "invalidate" and unknown.outcome["removed"] == [
        str(image / "jit" / f"{MODULE}.so")
    ]
    assert unknown.rebuild == once.Rebuild(left=(MODULE,), why="this card's target is unknown")


def test_two_sessions_at_once_build_it_once(image, monkeypatch) -> None:
    runner = Prewarm(wait=0.5)
    monkeypatch.setattr(once, "_run_prewarm", runner)
    start = threading.Barrier(2)
    results: list[once.ServingPreflight] = []

    def session() -> None:
        start.wait()
        results.append(once.prepare_serving_so_once({}, {}))

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
    monkeypatch.setattr(once, "_run_prewarm", runner)

    failed = once.prepare_serving_so_once({}, {})

    assert failed.outcome["action"] == "invalidate" and failed.rebuild.built == ()
    assert "compile_error" in failed.rebuild.why
    assert not (image / "jit" / f"{MODULE}.so").exists(), "moved aside, as before: the servers build it"
    assert "not built once" in failed.summary() and "each server builds them itself" in failed.summary()

    (image / "jit" / f"{MODULE}.so").write_bytes(module_of("a8w8_rowwise_retired"))
    again = once.prepare_serving_so_once({}, {})

    assert again.outcome["action"] == "invalidate" and len(runner.calls) == 1
    assert again.rebuild.why.startswith("tried once on this container already")


def test_a_build_that_does_not_cover_the_cards_rows_is_not_installed(image, monkeypatch) -> None:
    monkeypatch.setattr(once, "_run_prewarm", Prewarm(OTHER_GPU_KERNEL))

    result = once.prepare_serving_so_once({}, {})

    assert result.rebuild.built == () and "lack kernels of gfx950/256 rows of a8w8_tuned_gemm.csv" in result.rebuild.why
    assert not (image / "jit" / f"{MODULE}.so").exists()
    assert read_manifest(image / "jit") == {}


def test_a_module_of_a_table_the_boot_names_itself_is_not_built(image, monkeypatch, tmp_path) -> None:
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)
    own = tmp_path / "tuned.csv"
    own.write_text(f"gfx,cu_num,M,N,K,kernelName\n{GFX},{CU},1,2,3,a8w8_rowwise_tuned_here\n", encoding="utf-8")

    result = once.prepare_serving_so_once({VARIABLE: str(own)}, {})

    assert result.outcome["action"] == "invalidate" and runner.calls == []
    assert result.rebuild.why == "the boot names its own table for every module it moved aside"


def test_an_unknown_card_builds_nothing(image, monkeypatch) -> None:
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)
    monkeypatch.setattr(_aiter_jit, "card_target", lambda: None)

    result = once.prepare_serving_so_once({}, {})

    assert result.outcome["action"] == "invalidate" and runner.calls == []
    assert result.rebuild.why == "this card's target is unknown"


@pytest.mark.asyncio
async def test_the_launch_logs_the_decision_at_the_level_optimize_logs_at(image, monkeypatch, caplog) -> None:
    """``hyperloom optimize`` logs at WARNING without ``-v``: an INFO line would not reach its log."""
    monkeypatch.setattr(once, "_run_prewarm", Prewarm())
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
    assert f"built {MODULE} once from the image's own tables" in moved
    assert covered.endswith("the serving .so covers the CSVs this boot loads for gfx950/256; nothing moved")


#: A prewarm in place of the real one, run as the preflight runs it (``python -m``): it keeps its argv where
#: ``FAKE_PREWARM_ARGV`` says, logs as the prewarm does and reports that it built nothing.
_FAKE_PREWARM = """\
import json, os, sys
with open(os.environ["FAKE_PREWARM_ARGV"], "w", encoding="utf-8") as out:
    json.dump(sys.argv[1:], out)
print("INFO fake: a line that is not a warning", file=sys.stderr, flush=True)
print("WARNING fake: aiter prewarm: building module_gemm_a8w8 for gfx950/256 on CPU, about 180 s", file=sys.stderr)
print(json.dumps({"built": [], "present": [], "left": [], "failure": "the fake builds nothing", "seconds": 0.1}))
"""


def test_the_preflight_runs_the_prewarm_command_for_the_card_and_relays_its_warnings(
    image, monkeypatch, tmp_path, caplog
) -> None:
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "fake_prewarm.py").write_text(_FAKE_PREWARM, encoding="utf-8")
    monkeypatch.setattr(once, "PREWARM_MODULE", "fake_prewarm")
    monkeypatch.setenv("PYTHONPATH", str(fakes))
    monkeypatch.setenv("FAKE_PREWARM_ARGV", str(tmp_path / "argv.json"))
    caplog.set_level(logging.WARNING)

    result = once.prepare_serving_so_once({}, {})

    argv = json.loads((tmp_path / "argv.json").read_text(encoding="utf-8"))
    assert argv == ["--gfx", GFX, "--cu-num", str(CU), "--package", str(image), "--table", VARIABLE]
    assert result.rebuild == once.Rebuild(left=(MODULE,), seconds=0.1, why="the fake builds nothing")
    said = [record.getMessage() for record in caplog.records if record.name == once.log.name]
    assert (
        "aiter serving preflight: the prewarm says fake: aiter prewarm: building module_gemm_a8w8 for gfx950/256 on "
        "CPU, about 180 s"
    ) in said
    assert not any("not a warning" in line for line in said)


def test_a_prewarm_that_gives_no_report_is_a_failure_naming_its_last_words(image, monkeypatch, tmp_path) -> None:
    fakes = tmp_path / "fakes"
    fakes.mkdir()
    (fakes / "fake_prewarm.py").write_text("import sys\nsys.exit('the image has no compiler')\n", encoding="utf-8")
    monkeypatch.setattr(once, "PREWARM_MODULE", "fake_prewarm")
    monkeypatch.setenv("PYTHONPATH", str(fakes))

    result = once.prepare_serving_so_once({}, {})

    assert result.rebuild.why == "the prewarm exited 1 with no report: the image has no compiler"


def test_servers_that_build_for_several_targets_get_nothing_built_once(image, monkeypatch) -> None:
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)

    result = once.prepare_serving_so_once({}, {"GPU_ARCHS": f"{GFX};gfx942", "CU_NUM": str(CU)})

    assert result.outcome["targets"] == ["gfx942/256", "gfx950/256"] and runner.calls == []
    assert result.rebuild == once.Rebuild(left=(MODULE,), why="the servers build for 2 targets")


_BUILD_TARGETS_PY = 'GFX_CU_NUM_MAP = {"gfx942": 304, "gfx950": 256}\n'


@pytest.mark.parametrize(
    ("environ", "card", "boot"),
    [
        ({"CU_NUM": "304"}, (GFX, CU), "gfx950/304"),
        ({"GPU_ARCHS": "gfx942"}, ("gfx942", 256), "gfx942/304"),
    ],
    ids=["cu-num-304", "gpu-archs-gfx942-on-a-256-cu-card"],
)
def test_a_boot_whose_servers_build_for_another_target_than_the_card_gets_nothing_built_once(
    image, monkeypatch, environ, card, boot
) -> None:
    """Board #169 review: ``GPU_ARCHS=gfx942`` with no ``CU_NUM`` resolved to 304 CUs on a card of 256, and the module was
    built for 304. A test's server (which gets neither variable) builds for the card; the boot builds for its own."""
    (image / "jit" / "utils").mkdir(parents=True, exist_ok=True)
    (image / "jit" / "utils" / "build_targets.py").write_text(_BUILD_TARGETS_PY, encoding="utf-8")
    gfx, cu_num = boot.split("/")
    table = image / "configs" / "a8w8_tuned_gemm.csv"
    table.write_text(TABLE + f"{gfx},{cu_num},1,2,3,4,a8w8_rowwise_boot\n", encoding="utf-8")
    monkeypatch.setattr(_aiter_jit, "card_target", lambda: card)
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)

    result = once.prepare_serving_so_once({}, environ)

    assert result.outcome["action"] == "invalidate" and result.outcome["targets"] == [boot] and runner.calls == []
    assert result.rebuild.why == (
        f"this boot's servers build for {boot} (GPU_ARCHS or CU_NUM in its environment) and a test's for the card's "
        f"{card[0]}/{card[1]}"
    )


def test_serving_modules_outside_the_images_aiter_jit_are_not_built_into_it(image, monkeypatch, tmp_path) -> None:
    elsewhere = tmp_path / "elsewhere" / "jit"
    elsewhere.mkdir(parents=True)
    shutil.copyfile(image / "jit" / f"{MODULE}.so", elsewhere / f"{MODULE}.so")
    monkeypatch.setattr(_aiter_jit, "resolve_serving_context", lambda *args, **kwargs: (image, elsewhere / "build"))
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)

    result = once.prepare_serving_so_once({}, {})

    assert result.outcome["removed"] == [str(elsewhere / f"{MODULE}.so")] and runner.calls == []
    assert result.rebuild.why == f"the serving modules are not the image's own aiter/jit: {elsewhere}"
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of("a8w8_rowwise_retired")


@pytest.mark.asyncio
async def test_a_host_directory_that_cannot_be_used_leaves_the_preflight_to_run_unlocked(
    image, monkeypatch, tmp_path, caplog
) -> None:
    """The move protects the boot from a registry mismatch: a lock that cannot be had is no reason to skip it."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the directory would be", encoding="utf-8")
    monkeypatch.setattr(once, "HOST_DIR", blocker / "host")
    runner = Prewarm()
    monkeypatch.setattr(once, "_run_prewarm", runner)
    caplog.set_level(logging.WARNING)

    await launch_mod._prepare_aiter_serving_so({})

    assert not (image / "jit" / f"{MODULE}.so").exists(), "moved aside: the boot builds it"
    assert runner.calls == []
    lines = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any(
        line.startswith(
            f"aiter serving preflight: this host's lock {blocker / 'host' / 'preflight.lock'} cannot be used"
        )
        and line.endswith("the launch preflight of a boot runs unlocked")
        for line in lines
    )
    assert any("not built once (this host's lock cannot be used)" in line for line in lines)


def test_a_preflight_that_waits_for_the_lock_says_who_holds_it_and_what_it_does(image, monkeypatch, caplog) -> None:
    host = once.HOST_DIR
    host.mkdir(parents=True)
    holder = {"pid": 4242, "since": "2026-11-03T00:00:00Z", "doing": "building module_gemm_a8w8 for gfx950/256"}
    (host / "preflight.lock").write_text(json.dumps(holder), encoding="utf-8")
    flock = fcntl.flock

    def held_by_another(descriptor: int, operation: int) -> None:
        if operation & fcntl.LOCK_NB:
            raise BlockingIOError(11, "Resource temporarily unavailable")
        flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", held_by_another)
    monkeypatch.setattr(once, "_run_prewarm", Prewarm())
    caplog.set_level(logging.WARNING)

    once.prepare_serving_so_once({}, {})

    lines = [record.getMessage() for record in caplog.records if record.name == once.log.name]
    assert lines[0] == (
        f"aiter serving preflight: the launch preflight of a boot waits for this host's lock {host / 'preflight.lock'}, "
        "held by pid 4242 since 2026-11-03T00:00:00Z: building module_gemm_a8w8 for gfx950/256"
    )
    assert lines[1].startswith("aiter serving preflight: the launch preflight of a boot got this host's lock after")
    assert (host / "preflight.lock").read_text(encoding="utf-8") == "", "the holder's word goes with its lock"


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
    monkeypatch.setattr(once, "_run_prewarm", runner)
    caplog.set_level(logging.WARNING)

    once.prepare_serving_so_once({}, {})

    bound = once._PREWARM_TIMEOUT_SEC
    assert (
        f"aiter serving preflight: building {MODULE} once for {GFX}/{CU} before this boot (the prewarm of {VARIABLE}, "
        f"at most {bound} s)"
    ) in runner.said
    assert f"aiter prewarm: building {MODULE} for {GFX}/{CU} on CPU, about 65 s" in " ".join(runner.said), (
        "the stand-in's module is neither a CK-tile one nor generated by aiter's gen_instances: 65 s"
    )
    assert runner.holder["pid"] == os.getpid()
    assert runner.holder["doing"] == f"building {MODULE} for {GFX}/{CU} (the prewarm, at most {bound} s)"


def test_a_module_aiter_cannot_build_by_name_is_reported_as_left_to_the_servers(image, monkeypatch) -> None:
    config_file = image / "jit" / "optCompilerConfig.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config[MODULE]["third_party"] = ["'ck'"]
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(once, "_run_prewarm", Prewarm())

    result = once.prepare_serving_so_once({}, {})

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
    monkeypatch.setattr(once, "_run_prewarm", runner)
    once.prepare_serving_so_once({}, {})
    (image / "jit" / f"{MODULE}.so").write_bytes(module_of("a8w8_rowwise_retired"))

    monkeypatch.setattr(_aiter_jit, "card_target", lambda: (GFX, 128))
    again = once.prepare_serving_so_once({}, {})

    assert again.outcome["targets"] == [f"{GFX}/128"]
    assert [call[1:] for call in runner.calls] == [(GFX, CU), (GFX, 128)]
