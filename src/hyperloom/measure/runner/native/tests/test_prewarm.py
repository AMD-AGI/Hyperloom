# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The modules an image's own tables need and its ``aiter/jit`` lacks: which they are, the watchdog that covers building
them, and the prewarm that builds them on CPU and installs them into the image, over the stand-in aiter."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperloom.measure import farm as farm_module
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.jit.prewarmed import PREWARM_MANIFEST, read_manifest, unlike_prewarmed
from hyperloom.measure.jit.session import link_prebuilt
from hyperloom.measure.runner.cache_layout import file_sha256
from hyperloom.measure.runner.native import aiter_modules, prewarm
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import Isolation
from hyperloom.measure.runner.native.prewarm import (
    PrewarmError,
    isolated_builder,
    missing_prebuilt_modules,
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
    assert watchdog_seconds_for(["a", "b", "c", "d"]) == 4 * 600 + 300 == rebuild_watchdog_sec(["a", "b", "c", "d"])
    assert watchdog_seconds_for(["module_gemm_a8w8_bpreshuffle_cktile"]) == 2700 + 300, "a CK-tile module, 2700 s"


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


def test_the_cpu_build_merges_a_models_tables_below_its_own_tree_not_in_the_images_tmp(tmp_path, monkeypatch) -> None:
    rig = CpuRig(tmp_path, monkeypatch)
    model = rig.package / "configs" / "model_configs" / "qwen3_a8w8_tuned_gemm.csv"
    model.parent.mkdir()
    model.write_text("M,N,K\n7,8,9\n", encoding="utf-8")

    report = prewarm.prewarm(rig.package, rig.builder, gfx=GFX)

    assert not report.failure, "the image's /tmp/aiter_configs is not the build user's to write"
    assert report.built == (standin.MODULE,)


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


def _a8w8_image(root: Path, *, built_from: str) -> Path:
    """The stand-in aiter with its table module named as aiter's ``module_gemm_a8w8`` (``AITER_CONFIG_GEMM_A8W8``'s
    table), that module missing, and a table with a row for a gfx950 of 256 CUs and one for a gfx942 of 80."""
    package = standin.aiter_image(root)
    config_file = package / "jit" / "optCompilerConfig.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["module_gemm_a8w8"] = config.pop(standin.MODULE)
    config_file.write_text(json.dumps(config), encoding="utf-8")
    (package / "jit" / f"{standin.MODULE}.so").unlink()
    (package / "configs" / "a8w8_tuned_gemm.csv").write_text(
        f"gfx,cu_num,M,N,K,kernelName\ngfx950,256,1,2,3,{built_from}_card\ngfx942,80,1,2,3,{built_from}_other\n",
        encoding="utf-8",
    )
    return package


@pytest.mark.parametrize(("cu_num", "kernels", "installed"), [(256, "k_card", True), (256, "k_other", False)])
def test_a_build_for_a_cards_cu_count_is_installed_only_when_it_covers_that_cards_rows(
    tmp_path: Path, cu_num: int, kernels: str, installed: bool
) -> None:
    """aiter compiles the rows of the build's ``(gfx, CU_NUM)`` only; a module that lacks one of them is what the launch
    preflight moves aside again, so it is not installed. A row of another GPU's is no reason to refuse it."""
    package = _a8w8_image(tmp_path / "image", built_from="k")

    def builder(_package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        (scratch / "module_gemm_a8w8.so").write_bytes(b"\x7fELF " + kernels.encode())
        return {"module_gemm_a8w8": scratch / "module_gemm_a8w8.so"}

    report = prewarm.prewarm(package, builder, gfx=GFX, cu_num=cu_num, tables=["AITER_CONFIG_GEMM_A8W8"])

    jit = package / "jit"
    if installed:
        assert report.built == ("module_gemm_a8w8",) and not report.failure
        assert read_manifest(jit)["module_gemm_a8w8"]["cu_num"] == cu_num
    else:
        assert report.built == () and not (jit / "module_gemm_a8w8.so").exists()
        assert report.failure == (
            "PrewarmError: the built modules lack kernels of gfx950/256 rows of a8w8_tuned_gemm.csv"
        )


def test_the_command_builds_for_the_cards_cu_count_into_the_package_it_is_given(tmp_path, monkeypatch, capsys) -> None:
    package = _a8w8_image(tmp_path / "image", built_from="k")
    seen: dict[str, object] = {}
    monkeypatch.setenv("CU_NUM", "304")

    def builder_for(machine: BuildMachine, *, environ: Mapping[str, str] | None = None) -> prewarm.ModuleBuilder:
        seen.update(gfx=machine.gfx_arch, cu=(environ or {}).get("CU_NUM"))
        return StandIn()

    def located(name: str, python: str, cwd: Path) -> Path:
        raise AssertionError("--package names the package; nothing is located")

    monkeypatch.setattr(prewarm, "isolated_builder", builder_for)
    monkeypatch.setattr(prewarm, "locate_package", located)
    argv = ["--gfx", GFX, "--cu-num", "256", "--package", str(package), "--table", "AITER_CONFIG_GEMM_A8W8"]

    assert prewarm.main(argv) == 0

    printed = json.loads(capsys.readouterr().out)
    assert seen == {"gfx": GFX, "cu": "256"}
    assert printed["failure"].endswith("lack kernels of gfx950/256 rows of a8w8_tuned_gemm.csv"), (
        "the stand-in writes no kernel name, so the coverage check refuses it"
    )


def test_the_build_runs_under_what_a_tests_server_has_with_the_cards_cu_count() -> None:
    """A local runner's template carries the image's variables only (``local_templates.image_env``): a variable only the
    orchestrator has (``FLATMM_HIP_CLANG_PATH``, ``ENABLE_CK``) never reaches a test's server, so the build does not
    see it either; ``CU_NUM`` is the card's, ``GPU_ARCHS`` the build's own target."""
    orchestrator = {
        "FLATMM_HIP_CLANG_PATH": "/custom/llvm/bin",
        "ENABLE_CK": "0",
        "CC": "/usr/bin/gcc-13",
        "GPU_ARCHS": "gfx942",
        "CU_NUM": "304",
        "HIP_CLANG_PATH": "/opt/rocm/llvm/bin",
        "HIP_VISIBLE_DEVICES": "0,1",
        "AITER_LOG_MORE": "1",
        "ROCM_PATH": "/opt/rocm",
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "PATH": "/usr/bin",
    }

    assert prewarm.build_environment(orchestrator, 256) == {
        "HIP_CLANG_PATH": "/opt/rocm/llvm/bin",
        "AITER_LOG_MORE": "1",
        "ROCM_PATH": "/opt/rocm",
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "CU_NUM": "256",
    }
    assert "CU_NUM" not in prewarm.build_environment(orchestrator, None)


def test_the_command_builds_under_a_tests_environment_and_records_what_the_build_read(
    tmp_path, monkeypatch, capsys
) -> None:
    """Board #169 review: a ``*_HIP_CLANG_PATH`` set only in the orchestrator was baked into a module every test linked.
    The module is built without it, and a test whose server sets a variable its build read otherwise builds its own."""
    package = _a8w8_image(tmp_path / "image", built_from="k")
    config_file = package / "jit" / "optCompilerConfig.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config["module_gemm_a8w8"]["hip_clang_path"] = "os.environ.get('FLATMM_HIP_CLANG_PATH')"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    for name, value in {"FLATMM_HIP_CLANG_PATH": "/custom/llvm/bin", "ENABLE_CK": "0", "AITER_LOG_MORE": "1"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CU_NUM", raising=False)
    built_under: list[Mapping[str, str]] = []

    def builder_for(machine: BuildMachine, *, environ: Mapping[str, str] | None = None) -> prewarm.ModuleBuilder:
        built_under.append(dict(environ or {}))

        def build(_package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
            (scratch / "module_gemm_a8w8.so").write_bytes(b"\x7fELF k_card")
            return {"module_gemm_a8w8": scratch / "module_gemm_a8w8.so"}

        return build

    monkeypatch.setattr(prewarm, "isolated_builder", builder_for)
    argv = ["--gfx", GFX, "--cu-num", "256", "--package", str(package), "--table", "AITER_CONFIG_GEMM_A8W8"]

    assert prewarm.main(argv) == 0

    assert json.loads(capsys.readouterr().out)["built"] == ["module_gemm_a8w8"]
    [environ] = built_under
    assert environ["CU_NUM"] == "256" and environ["AITER_LOG_MORE"] == "1"
    assert "FLATMM_HIP_CLANG_PATH" not in environ and "ENABLE_CK" not in environ
    entry = read_manifest(package / "jit")["module_gemm_a8w8"]
    assert {"FLATMM_HIP_CLANG_PATH", "ENABLE_CK", "GPU_ARCHS"} <= set(entry["build_variables"])
    assert entry["build_environment"] == {} and entry["cu_num"] == 256 and entry["gfx"] == GFX
    untouched = {"AITER_LOG_MORE": "1", "ROCM_PATH": "/opt/rocm"}
    assert unlike_prewarmed(package / "jit", [untouched], GFX) == frozenset()
    orchestrators = {**untouched, "FLATMM_HIP_CLANG_PATH": "/custom/llvm/bin"}
    assert unlike_prewarmed(package / "jit", [orchestrators], GFX) == {"module_gemm_a8w8"}


def test_a_test_that_starts_while_a_module_is_installed_links_none_or_the_whole_module(tmp_path, monkeypatch) -> None:
    """The copy into the image's ``aiter/jit`` is half written when a test's start lists the directory (the hook runs
    the linker then): it must find no module of that name, never the half."""
    package = image(tmp_path / "image")
    whole = b"\x7fELF built:" + standin.MODULE.encode() + b" " + b"k" * 4096
    copy = prewarm.shutil.copyfile
    seen: list[bytes | None] = []

    def copy_with_a_test_starting_halfway(source: Path, target: Path) -> None:
        data = Path(source).read_bytes()
        Path(target).write_bytes(data[: len(data) // 2])
        content = tmp_path / f"test-{len(seen)}"
        content.mkdir()
        link_prebuilt(package / "jit", content, frozenset())
        linked = content / f"{standin.MODULE}.so"
        seen.append(linked.read_bytes() if linked.exists() else None)
        copy(source, target)

    def builder(_package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        built = {name: scratch / f"{name}.so" for name in modules}
        for name, path in built.items():
            path.write_bytes(whole if name == standin.MODULE else b"\x7fELF built:" + name.encode())
        return built

    monkeypatch.setattr(prewarm.shutil, "copyfile", copy_with_a_test_starting_halfway)
    report = prewarm.prewarm(package, builder, gfx=GFX)

    assert report.built == (standin.MODULE, BLOCKSCALE) and len(seen) == 2
    assert all(found in (None, whole) for found in seen), "a test linked a half-written module"
    assert (package / "jit" / f"{standin.MODULE}.so").read_bytes() == whole


def test_the_build_says_what_it_builds_for_which_card_and_how_long_before_it_starts(tmp_path, caplog) -> None:
    package = image(tmp_path / "image")
    said_before: list[list[str]] = []

    def builder(_package: Path, modules: frozenset[str], scratch: Path) -> Mapping[str, Path]:
        said_before.append([record.getMessage() for record in caplog.records if record.levelno == logging.WARNING])
        return StandIn()(_package, modules, scratch)

    caplog.set_level(logging.WARNING)
    prewarm.prewarm(package, builder, gfx=GFX, cu_num=256)

    [said] = said_before
    assert said == [
        f"aiter prewarm: building {standin.MODULE}, {BLOCKSCALE} for {GFX}/256 on CPU, about 130 s "
        "(each module's typical build on a card)"
    ]


def test_without_a_cu_count_the_command_takes_the_one_cu_num_names(tmp_path, monkeypatch, capsys) -> None:
    package = _a8w8_image(tmp_path / "image", built_from="k")
    seen: list[str | None] = []

    def builder_for(machine: BuildMachine, *, environ: Mapping[str, str] | None = None) -> prewarm.ModuleBuilder:
        seen.append((environ or {}).get("CU_NUM"))
        return StandIn()

    monkeypatch.setattr(prewarm, "isolated_builder", builder_for)
    monkeypatch.setenv("CU_NUM", "32")
    assert prewarm.main(["--gfx", GFX, "--package", str(package)]) == 0
    assert seen == ["32"] and json.loads(capsys.readouterr().out)["built"] == ["module_gemm_a8w8"]
    assert read_manifest(package / "jit")["module_gemm_a8w8"]["cu_num"] == 32
