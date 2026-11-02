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

import json
import logging
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
    monkeypatch.setattr(_aiter_jit, "_card_target", lambda: (GFX, CU))
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
    monkeypatch.setattr(_aiter_jit, "_card_target", lambda: None)
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
    entry = prewarm.read_manifest(image / "jit")[MODULE]
    assert entry["gfx"] == GFX and entry["cu_num"] == CU
    assert "built module_gemm_a8w8 once from the image's own tables" in first.summary()

    again = once.prepare_serving_so_once({}, {})
    assert again.outcome["action"] == "skip" and again.rebuild is None and len(runner.calls) == 1

    monkeypatch.setattr(_aiter_jit, "_card_target", lambda: None)
    unknown = once.prepare_serving_so_once({}, {})
    assert unknown.outcome["action"] == "skip" and unknown.outcome["kept_image_own"] == [MODULE], (
        "with every row counting, the prewarm's manifest keeps it"
    )
    assert (image / "jit" / f"{MODULE}.so").read_bytes() == module_of(CARD_KERNEL)
    assert unknown.summary().endswith(f"but in {MODULE}, which the prewarm built from these same tables")


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
    assert prewarm.read_manifest(image / "jit") == {}


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
    monkeypatch.setattr(_aiter_jit, "_card_target", lambda: None)

    result = once.prepare_serving_so_once({}, {})

    assert result.outcome["action"] == "invalidate" and runner.calls == []
    assert result.rebuild.why == "this card's target is unknown"


@pytest.mark.asyncio
async def test_the_launch_logs_the_decision_where_prelude_s_child_logs_it(image, monkeypatch, caplog) -> None:
    """``hyperloom optimize`` logs at WARNING without ``-v``: an INFO line never reached PRELUDE's log."""
    monkeypatch.setattr(once, "_run_prewarm", Prewarm())
    monkeypatch.setattr(launch_mod, "_COVERED_SAID", set())
    caplog.set_level(logging.WARNING)

    await launch_mod._prepare_aiter_serving_so({})
    await launch_mod._prepare_aiter_serving_so({})
    await launch_mod._prepare_aiter_serving_so({})

    lines = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(lines) == 2, "a move every time, nothing moved once"
    moved, covered = lines
    assert "outran the serving .so for gfx950/256; moved 1 module(s) aside" in moved
    assert f"built {MODULE} once from the image's own tables" in moved
    assert covered.endswith("the serving .so covers the CSVs this boot loads for gfx950/256; nothing moved")
