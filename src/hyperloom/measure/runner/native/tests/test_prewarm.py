# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The modules an image's own tables need and its ``aiter/jit`` lacks: which they are, the watchdog that covers building
them, and the prewarm that builds them on CPU and installs them into the image, over the stand-in aiter."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperloom.measure import farm as farm_module
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.cache_layout import file_sha256
from hyperloom.measure.runner.native import aiter_modules, prewarm
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import Isolation
from hyperloom.measure.runner.native.prewarm import (
    PREWARM_MANIFEST,
    PrewarmError,
    isolated_builder,
    missing_prebuilt_modules,
    read_manifest,
    watchdog_seconds_for,
)
from hyperloom.measure.runner.native.tests import _aiter_standin as standin
from hyperloom.measure.runner.rebuild_watchdog import rebuild_watchdog_sec

GFX = "gfx950"
#: A second table module, generated from another variable's table.
BLOCKSCALE = "module_gemm_blockscale"
BLOCKSCALE_VARIABLE = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
#: A table module aiter cannot build by name (it fetches a third-party checkout): left to the launch.
FETCHING = "module_gemm_fetching"


def image(root: Path, *, missing: tuple[str, ...] = (standin.MODULE, BLOCKSCALE, FETCHING)) -> Path:
    """The stand-in aiter with two more table modules; ``missing`` are not in its ``aiter/jit``."""
    package = standin.aiter_image(root)
    config = json.loads((package / "jit" / "optCompilerConfig.json").read_text(encoding="utf-8"))
    generator = standin.GENERATOR.replace(standin.TABLE_VARIABLE, BLOCKSCALE_VARIABLE)
    config[BLOCKSCALE] = {**standin.entry("k.cu"), "blob_gen_cmd": generator}
    config[FETCHING] = {**config[standin.MODULE], "third_party": ["'ck'"]}
    (package / "jit" / "optCompilerConfig.json").write_text(json.dumps(config), encoding="utf-8")
    for name in (BLOCKSCALE, FETCHING):
        (package / "jit" / f"{name}.so").write_text("image build", encoding="utf-8")
    for name in missing:
        (package / "jit" / f"{name}.so").unlink()
    return package


class StandIn:
    """A builder that writes ``built:<name>`` for every module it is asked for, or fails as told."""

    def __init__(self, *, fail: Exception | None = None, skip: frozenset[str] = frozenset()) -> None:
        self.calls: list[frozenset[str]] = []
        self.fail, self.skip = fail, skip

    def __call__(self, package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        self.calls.append(modules)
        if self.fail is not None:
            raise self.fail
        built = {}
        for name in sorted(modules - self.skip):
            built[name] = scratch / f"{name}.so"
            built[name].write_bytes(b"\x7fELF built:" + name.encode())
        return built


def jit_listing(package: Path) -> dict[str, bytes]:
    jit = package / "jit"
    return {path.name: path.read_bytes() for path in sorted(jit.iterdir()) if path.is_file()}


def test_the_table_modules_the_image_lacks_are_missing_and_the_rest_are_not(tmp_path: Path) -> None:
    package = image(tmp_path / "image")
    (package / "jit" / f"{standin.OTHER}.so").unlink()

    assert missing_prebuilt_modules(package) == [standin.MODULE, BLOCKSCALE, FETCHING]
    assert missing_prebuilt_modules(package, tables=[standin.TABLE_VARIABLE]) == [standin.MODULE, FETCHING]
    assert missing_prebuilt_modules(package, tables=[BLOCKSCALE_VARIABLE]) == [BLOCKSCALE]
    assert missing_prebuilt_modules(image(tmp_path / "full", missing=())) == []


def test_an_image_whose_module_config_cannot_be_read_says_so_rather_than_nothing_missing(tmp_path: Path) -> None:
    package = image(tmp_path / "image")
    (package / "jit" / "optCompilerConfig.json").unlink()

    with pytest.raises(FileNotFoundError):
        missing_prebuilt_modules(package)


def test_the_watchdog_covers_each_module_build_plus_sglangs_own() -> None:
    assert watchdog_seconds_for([]) == 300
    assert watchdog_seconds_for(["a", "b", "c", "d"]) == 4 * 600 + 300 == rebuild_watchdog_sec(4)


def test_prewarm_builds_what_is_missing_installs_it_with_a_manifest_and_a_rerun_does_nothing(tmp_path: Path) -> None:
    package = image(tmp_path / "image")
    builder = StandIn()

    report = prewarm.prewarm(package, builder, gfx=GFX)

    assert builder.calls == [frozenset({standin.MODULE, BLOCKSCALE})], "only what is missing and buildable by name"
    assert report.built == (standin.MODULE, BLOCKSCALE) and report.left == (FETCHING,) and not report.failure
    assert report.present == ()
    jit = package / "jit"
    assert (jit / f"{standin.MODULE}.so").read_bytes() == b"\x7fELF built:" + standin.MODULE.encode()
    assert not (jit / f"{FETCHING}.so").exists(), "a module aiter cannot build by name is left to the launch"
    entries = read_manifest(jit)
    assert set(entries) == {standin.MODULE, BLOCKSCALE}
    for name, entry in entries.items():
        assert entry["sha256"] == file_sha256(jit / f"{name}.so")
        assert entry["gfx"] == GFX and entry["built_at"].endswith("Z")
        assert entry["tables_sha256"] == prewarm.tables_digest(package)
        assert entry["source_digest"] == prewarm.source_digest(package, entry["tables_sha256"])
    assert report.manifest == str(jit / PREWARM_MANIFEST)
    assert not [path.name for path in jit.iterdir() if path.name.startswith(".")], "no temporary file is left"

    before = jit_listing(package)
    again = prewarm.prewarm(package, builder, gfx=GFX)

    assert len(builder.calls) == 1, "a rerun builds nothing"
    assert again.built == () and again.present == (standin.MODULE, BLOCKSCALE) and again.left == (FETCHING,)
    assert jit_listing(package) == before


def test_a_rerun_after_one_more_module_went_missing_keeps_the_earlier_manifest_entries(tmp_path: Path) -> None:
    package = image(tmp_path / "image", missing=(standin.MODULE,))
    builder = StandIn()
    prewarm.prewarm(package, builder, gfx=GFX)
    (package / "jit" / f"{BLOCKSCALE}.so").unlink()

    report = prewarm.prewarm(package, builder, gfx=GFX)

    assert builder.calls[-1] == frozenset({BLOCKSCALE}) and report.built == (BLOCKSCALE,)
    assert set(read_manifest(package / "jit")) == {standin.MODULE, BLOCKSCALE}


@pytest.mark.parametrize(
    ("builder", "named"),
    [
        (StandIn(fail=BuildInfraError("the builder failed its preflight")), "BuildInfraError"),
        (StandIn(fail=PrewarmError("the build failed: compile_error")), "compile_error"),
        (StandIn(skip=frozenset({BLOCKSCALE})), f"the build made no {BLOCKSCALE}"),
    ],
    ids=["machine", "compile", "incomplete"],
)
def test_a_failed_build_leaves_the_image_unchanged_and_says_why(tmp_path: Path, builder: StandIn, named: str) -> None:
    package = image(tmp_path / "image")
    before = jit_listing(package)

    report = prewarm.prewarm(package, builder, gfx=GFX)

    assert named in report.failure and report.built == ()
    assert jit_listing(package) == before


def test_an_install_that_fails_part_way_leaves_no_module_and_no_temporary_file(tmp_path: Path, monkeypatch) -> None:
    package = image(tmp_path / "image")
    before = jit_listing(package)
    copy = prewarm.shutil.copyfile

    def full_disk(source: Path, target: Path) -> None:
        if Path(source).name == f"{BLOCKSCALE}.so":
            raise OSError(28, "No space left on device")
        copy(source, target)

    monkeypatch.setattr(prewarm.shutil, "copyfile", full_disk)
    report = prewarm.prewarm(package, StandIn(), gfx=GFX)

    assert report.failure == "OSError: [Errno 28] No space left on device" and report.built == ()
    assert jit_listing(package) == before, "the module copied before the failure is not installed either"


def test_the_tables_digest_changes_with_a_table_and_the_source_digest_with_csrc(tmp_path: Path) -> None:
    package = image(tmp_path / "image")
    tables = prewarm.tables_digest(package)
    source = prewarm.source_digest(package, tables)

    (package.parent / standin.KERNEL).write_text("// changed\n", encoding="utf-8")
    assert prewarm.tables_digest(package) == tables and prewarm.source_digest(package, tables) != source
    (package.parent / standin.TABLE).write_text("M,N,K\n4,5,6\n", encoding="utf-8")
    assert prewarm.tables_digest(package) != tables


def test_a_manifest_of_another_schema_lists_nothing(tmp_path: Path) -> None:
    (tmp_path / PREWARM_MANIFEST).write_text(json.dumps({"schema": "other/1", "modules": [{"name": "m"}]}), "utf-8")
    assert read_manifest(tmp_path) == {}
    (tmp_path / PREWARM_MANIFEST).write_text("not json", "utf-8")
    assert read_manifest(tmp_path) == {}


class CpuRig:
    """The CPU build runner of #125 stage 2b over the stand-in aiter (no user change: the test's own)."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch, located: Path | None = None) -> None:
        self.package = standin.aiter_image(tmp / "image")
        (self.package / "jit" / f"{standin.MODULE}.so").unlink()
        found = located or self.package
        locate = lambda name, python, cwd: found.parent / name  # noqa: E731
        for module in (aiter_modules, farm_module, prewarm):
            monkeypatch.setattr(module, "locate_package", locate)
        machine = BuildMachine("prewarm", "prewarm", GFX, sys.executable)
        self.builder = isolated_builder(
            machine, isolation=lambda: Isolation((), None), factory=standin.FACTORY, environ={}
        )


def test_the_cpu_build_runner_builds_the_missing_modules_from_the_images_own_tree(tmp_path, monkeypatch) -> None:
    rig = CpuRig(tmp_path, monkeypatch)

    report = prewarm.prewarm(rig.package, rig.builder, gfx=GFX)

    assert report.built == (standin.MODULE,) and not report.failure
    built = (rig.package / "jit" / f"{standin.MODULE}.so").read_bytes()
    assert built == standin.built(GFX, standin.IMAGE_KERNEL), "built for the target from the image's k.cu"
    assert set(read_manifest(rig.package / "jit")) == {standin.MODULE}


def test_a_compile_error_of_the_cpu_build_installs_nothing_and_names_the_error(tmp_path, monkeypatch) -> None:
    rig = CpuRig(tmp_path, monkeypatch)
    (rig.package.parent / standin.KERNEL).write_text("#error the image's tree does not compile\n", encoding="utf-8")
    before = jit_listing(rig.package)

    report = prewarm.prewarm(rig.package, rig.builder, gfx=GFX)

    assert report.failure.startswith("PrewarmError: the build failed: compile_error: the compile exited 1")
    assert "the image's tree does not compile" in report.failure
    assert jit_listing(rig.package) == before


def test_the_cpu_build_runner_refuses_an_interpreter_that_imports_another_aiter(tmp_path, monkeypatch) -> None:
    other = standin.aiter_image(tmp_path / "other")
    rig = CpuRig(tmp_path, monkeypatch, located=other)
    before = jit_listing(rig.package)

    report = prewarm.prewarm(rig.package, rig.builder, gfx=GFX)

    assert report.failure == f"PrewarmError: {sys.executable} imports aiter from {other}, not from {rig.package}"
    assert jit_listing(rig.package) == before


def test_the_command_needs_the_target_and_reports_an_image_without_aiter_without_failing(
    monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("HYPERLOOM_BUILD_GFX", "GPU_ARCHS"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="prewarm needs --gfx"):
        prewarm.main([])

    def no_aiter(name: str, python: str, cwd: Path) -> Path:
        raise farm_module.FarmError("no module named aiter")

    monkeypatch.setattr(prewarm, "locate_package", no_aiter)
    assert prewarm.main(["--gfx", GFX]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["failure"] == "the image's aiter cannot be found: no module named aiter" and printed["built"] == []
