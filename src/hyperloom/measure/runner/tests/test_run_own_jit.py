# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test the compiled-code cache gives no aiter JIT directory builds its modules in one of its own, not in the image.

The image (``test_run_jit_tables``) ships ``module_gemm`` and ``module_other`` built. The server is a fake grid that
does what aiter's JIT does: it looks for each module in ``AITER_JIT_DIR``, else in the ``jit`` directory of the package
(the image's), and builds the one it does not find (``module_new``) there. Before the fix a test with no patch and no
cache had no farm and no session, so its build was written into the image's ``aiter/jit``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import yaml

from hyperloom.measure import local
from hyperloom.measure.farm import Farm
from hyperloom.measure.runner.own_jit import OWN_AITER_JIT_DIR, own_aiter_jit
from hyperloom.measure.runner.tests.test_run_jit import (  # noqa: F401 - fixtures
    TABLE_VARIABLE,
    Rig,
    _arm,
    env,
    key,
    rig,
)
from hyperloom.measure.runner.tests.test_run_jit_tables import (  # noqa: F401 - fixtures
    IMAGE_MODULE,
    TEST_TABLE,
    _by_patch,
    _by_variable,
    _deliver,
    image,
)
from hyperloom.measure.runner.tests.test_run_jit_variables import ARCH_FLAG, FLAG, GFX, VARIABLE
from hyperloom.measure.socket import TestCall

BUILT = b"\x7fELF built by the test"
PREWARMED = b"\x7fELF built once by the launch preflight"
WANTED = ("module_gemm", "module_other", "module_new")


class _JitServer:
    """What the server's aiter does with its JIT directory; what it found, where it built and what the directory held
    once it had built (each module a link or not, and its bytes), by test: the runner removes the directory afterwards."""

    def __init__(self, rig: Rig, monkeypatch) -> None:
        self.image_jit = rig.bases / "aiter" / "jit"
        self.found: dict[str, bool] = {}
        self.held: dict[str, tuple[bool, bytes]] = {}
        self.jit_dir: Path | None = None
        grid = local.run_grid

        async def serve(**kwargs):
            jit = Path(os.environ.get("AITER_JIT_DIR") or self.image_jit)
            self.jit_dir = jit
            for module in WANTED:
                target = jit / f"{module}.so"
                self.found[module] = target.exists()
                if not target.exists():
                    target.write_bytes(BUILT)
            self.held = {path.name: (path.is_symlink(), path.read_bytes()) for path in jit.iterdir()}
            return await grid(**kwargs)

        monkeypatch.setattr(local, "run_grid", serve)


def _image_jit(rig: Rig) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted((rig.bases / "aiter" / "jit").iterdir())}


def _run(rig: Rig, monkeypatch, name: str, make_sheet, **kwargs) -> tuple[_JitServer, Path]:
    before = _image_jit(rig)
    original = rig._start

    def start(*args, **start_kwargs):
        run, environ = original(*args, **start_kwargs)
        server.__init__(rig, monkeypatch)
        return run, environ

    server = _JitServer.__new__(_JitServer)
    monkeypatch.setattr(rig, "_start", start)
    _, _, run = rig.run(name, make_sheet, **kwargs)
    assert _image_jit(rig) == before, "nothing the server built reached the image's aiter/jit"
    assert list(rig.host_tmp.iterdir()) == [], "the test's compile directory went with the test"
    return server, run


def _in_a_compile_directory(rig: Rig, compile_dir: Path) -> bool:
    """Whether ``compile_dir`` is a test's compile directory on the host's temporary disk, not in its sandbox."""
    return compile_dir.parent == rig.host_tmp and compile_dir.name.startswith("hl-jit-")


def test_a_test_with_no_patch_and_no_cache_builds_in_its_own_jit_directory_beside_the_images_modules(
    image, monkeypatch
) -> None:
    server, run = _run(image, monkeypatch, "plain", _arm(image), cache=False)

    assert server.jit_dir.name == OWN_AITER_JIT_DIR and _in_a_compile_directory(image, server.jit_dir.parent)
    assert not (run / "scratch" / "test" / OWN_AITER_JIT_DIR).exists(), "nothing is built on the sandbox's disk"
    assert server.found == {"module_gemm": True, "module_other": True, "module_new": False}
    assert server.held["module_gemm.so"] == (True, IMAGE_MODULE.encode())
    assert server.held["module_new.so"] == (False, BUILT)


def test_a_module_the_tests_table_makes_stale_is_not_linked_but_built(image, monkeypatch) -> None:
    server, _ = _run(image, monkeypatch, "table", _deliver(image, _by_variable(TEST_TABLE)), cache=False)

    assert server.found == {"module_gemm": False, "module_other": True, "module_new": False}
    assert server.held["module_gemm.so"] == (False, BUILT)


def test_a_jit_directory_the_sheet_names_is_left_as_it_is(image, monkeypatch, tmp_path) -> None:
    named = tmp_path / "named_jit"
    named.mkdir()
    server, run = _run(image, monkeypatch, "named", _arm(image, process_env={"AITER_JIT_DIR": str(named)}), cache=False)

    assert server.jit_dir == named and not (run / "scratch" / "test" / OWN_AITER_JIT_DIR).exists()


def test_a_test_the_cache_gives_a_jit_directory_gets_no_second_one(image, monkeypatch) -> None:
    server, run = _run(image, monkeypatch, "cached", _arm(image), cache=True)

    assert server.jit_dir.name == "aiter_jit" and server.jit_dir.parent.name == "jit", "the cache's directory"
    assert _in_a_compile_directory(image, server.jit_dir.parent.parent)
    assert not (run / "scratch" / "test" / "jit").exists(), "nothing is built on the sandbox's disk"
    assert not (run / "scratch" / "test" / OWN_AITER_JIT_DIR).exists()


def test_a_jit_directory_a_points_server_environment_names_is_left_as_it_is(image, tmp_path) -> None:
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"envs": {"AITER_JIT_DIR": "/elsewhere"}}}), encoding="utf-8")
    call = TestCall(
        points=(point,), output_dir=tmp_path, timeout_sec=1, cwd=str(tmp_path), magpie_python="", task_id="t"
    )

    assert own_aiter_jit(call, None, None, {}, tmp_path / OWN_AITER_JIT_DIR) == {}
    plain = dataclasses.replace(call, points=(_plain_point(tmp_path),))
    assert own_aiter_jit(plain, None, None, {}, tmp_path / OWN_AITER_JIT_DIR) == {
        "AITER_JIT_DIR": str(tmp_path / OWN_AITER_JIT_DIR)
    }
    assert own_aiter_jit(plain, None, None, {"AITER_JIT_DIR": "/inherited"}, tmp_path / "other") == {}
    rebuilt = Farm(tmp_path / "farm" / "site", ("aiter",))
    assert own_aiter_jit(plain, rebuilt, None, {}, tmp_path / "farm_test") == {}, "the farm's jit is the test's own"
    other = Farm(tmp_path / "farm" / "site", ("sglang",))
    assert own_aiter_jit(plain, other, None, {}, tmp_path / "sglang_test") != {}, "the server imports the image's aiter"


def _plain_point(tmp_path: Path) -> Path:
    point = tmp_path / "plain.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"envs": {"TP": 1}}}), encoding="utf-8")
    return point


def test_a_farm_with_no_cache_leaves_out_the_module_its_patched_table_makes_stale(image, monkeypatch) -> None:
    """A configs patch rebuilds aiter in the farm, whose ``jit`` links the image's modules. With no cache to leave the
    stale one out, the server loaded the image's module and stopped on a kernel 'not present in the compiled registry'."""
    _, _, run = image.run("patched", _deliver(image, _by_patch(image, TEST_TABLE)), cache=False)

    farm_jit = next(run.rglob("farm/site/aiter/jit"))
    assert (farm_jit / "module_other.so").is_symlink(), "a module the table does not feed stays the image's"
    assert not os.path.lexists(farm_jit / "module_gemm.so"), "the stale module is built from the farm's table"
    assert (image.bases / "aiter" / "jit" / "module_gemm.so").read_text() == IMAGE_MODULE


def _module_gemm_reads_a_build_variable(rig: Rig, flag: str = FLAG) -> None:
    """``module_gemm``'s entry passes a flag aiter takes from the environment as it builds (``test_run_jit_variables``)."""
    path = rig.bases / "aiter" / "jit" / "optCompilerConfig.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["module_gemm"]["flags_extra_hip"] = [flag]
    path.write_text(json.dumps(config), encoding="utf-8")


def test_a_module_a_build_variable_of_the_test_makes_stale_is_not_linked_but_built(image, monkeypatch) -> None:
    """With no cache the server would otherwise load the image's module, built under the image's value."""
    _module_gemm_reads_a_build_variable(image)

    server, _ = _run(image, monkeypatch, "variable", _arm(image, envs={VARIABLE: "3"}), cache=False)

    assert server.found == {"module_gemm": False, "module_other": True, "module_new": False}
    assert server.held["module_gemm.so"] == (False, BUILT)


def test_a_farm_with_no_cache_leaves_out_the_module_a_build_variable_makes_stale(image) -> None:
    _module_gemm_reads_a_build_variable(image)

    _, _, run = image.run(
        "farm-variable", _arm(image, envs={VARIABLE: "3"}, roots=_aiter_package_root(image)), cache=False
    )

    farm_jit = next(run.rglob("farm/site/aiter/jit"))
    assert (farm_jit / "module_other.so").is_symlink()
    assert not os.path.lexists(farm_jit / "module_gemm.so"), "built under the test's value, not linked"


_ARCHS = [("native", True), (GFX, True), ("gfx942", False), (f"gfx942;{GFX}", False)]


@pytest.mark.parametrize(("value", "image_kept"), _ARCHS)
def test_gpu_archs_naming_the_card_leaves_the_images_module_linked_with_no_cache(
    image, monkeypatch, value, image_kept
) -> None:
    """The image's module was built for the card, as one built under ``native`` is: only another value makes it stale."""
    _module_gemm_reads_a_build_variable(image, ARCH_FLAG)

    server, _ = _run(image, monkeypatch, "arch", _arm(image, envs={"GPU_ARCHS": value}), cache=False)

    assert server.found["module_gemm"] is image_kept
    assert server.held["module_gemm.so"] == ((True, IMAGE_MODULE.encode()) if image_kept else (False, BUILT))


@pytest.mark.parametrize(("value", "image_kept"), _ARCHS)
def test_gpu_archs_naming_the_card_leaves_the_farms_link_to_the_images_module_with_no_cache(
    image, value, image_kept
) -> None:
    _module_gemm_reads_a_build_variable(image, ARCH_FLAG)

    _, _, run = image.run(
        "farm-arch", _arm(image, envs={"GPU_ARCHS": value}, roots=_aiter_package_root(image)), cache=False
    )

    farm_jit = next(run.rglob("farm/site/aiter/jit"))
    assert (farm_jit / "module_gemm.so").is_symlink() is image_kept
    assert os.path.lexists(farm_jit / "module_gemm.so") is image_kept


def _aiter_package_root(rig: Rig):
    return lambda run: [rig.root(run, "aiter", "ops/mha.py", b"X = 2\n")]


@pytest.mark.parametrize("cache", [False, True], ids=["own-directory", "cache-directory"])
@pytest.mark.parametrize("touched", [False, True], ids=["untouched", "touched-by-its-table"])
def test_a_module_built_once_into_the_image_is_linked_by_a_later_test_it_does_not_touch(
    image, monkeypatch, cache: bool, touched: bool
) -> None:
    """The launch preflight moved ``module_gemm`` aside and built it once into the image's ``aiter/jit``, replacing the
    file in place (board #169). A test that starts after links it, unless its own table touches it."""
    _prewarmed(image)
    sheet = _deliver(image, _by_variable(TEST_TABLE)) if touched else _arm(image)

    server, _ = _run(image, monkeypatch, "after", sheet, cache=cache)

    assert server.found["module_gemm"] is not touched
    assert server.held["module_gemm.so"] == ((False, BUILT) if touched else (True, PREWARMED))


#: What hyperloom's prewarm records of ``module_gemm``'s build (``measure.jit.prewarmed.build_record``): the variables the
#: image's aiter reads to build it, as aiter e7d2453f's would (its ``jit`` code reads ``CU_NUM`` and ``GPU_ARCHS``, a CK-tile
#: entry ``FLATMM_HIP_CLANG_PATH``), none of which the orchestrator's image variables set; built for the rig's card.
_RECORD = {
    "gfx": GFX,
    "cu_num": 256,
    "build_variables": ["AITER_META_DIR", "CU_NUM", "ENABLE_CK", "FLATMM_HIP_CLANG_PATH", "GPU_ARCHS"],
    "build_environment": {},
}


def _prewarmed(rig: Rig, **record: object) -> None:
    """The launch preflight moved ``module_gemm`` aside and built it once into the image's ``aiter/jit``, replacing the
    file in place, and listed it in the prewarm's manifest with ``record`` (board #169)."""
    jit = rig.bases / "aiter" / "jit"
    (jit / "module_gemm.so").unlink()
    installing = jit / ".module_gemm.so.prewarm"
    installing.write_bytes(PREWARMED)
    os.replace(installing, jit / "module_gemm.so")
    entry = {"name": "module_gemm", "sha256": "0" * 64, **(record or _RECORD)}
    document = {"schema": "hyperloom.aiter_prewarm/1", "modules": [entry]}
    (jit / "hyperloom-prewarm.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("cache", [False, True], ids=["own-directory", "cache-directory"])
@pytest.mark.parametrize(
    "envs",
    [{"CU_NUM": "304"}, {"GPU_ARCHS": "gfx942"}, {"FLATMM_HIP_CLANG_PATH": "/custom/llvm/bin"}, {"ENABLE_CK": "0"}],
    ids=["cu-num-304", "gpu-archs-gfx942", "a-clang-only-the-orchestrator-had", "enable-ck-0"],
)
def test_a_module_built_once_is_not_given_to_a_test_whose_server_would_build_another(
    image, monkeypatch, cache: bool, envs: dict[str, str]
) -> None:
    """Board #169 review: the module was built under one value of a variable its build reads, and a test whose server
    sets another (no module config entry of the rig reads any of these) loaded it."""
    _prewarmed(image)

    server, _ = _run(image, monkeypatch, "unlike", _arm(image, envs=envs), cache=cache)

    assert server.found["module_gemm"] is False and server.held["module_gemm.so"] == (False, BUILT)
    assert server.held["module_other.so"] == (True, IMAGE_MODULE.encode()), "a module the prewarm did not build is"


@pytest.mark.parametrize("cache", [False, True], ids=["own-directory", "cache-directory"])
@pytest.mark.parametrize(
    "record",
    [
        {**_RECORD, "build_environment": {"FLATMM_HIP_CLANG_PATH": "/custom/llvm/bin"}},
        {**_RECORD, "gfx": "gfx942"},
        {"gfx": GFX, "cu_num": 256},
    ],
    ids=["built-under-a-value-the-runner-has-not", "built-for-another-card", "no-record-of-its-variables"],
)
def test_a_module_built_once_otherwise_than_the_runners_servers_build_is_not_given(
    image, monkeypatch, cache: bool, record: dict
) -> None:
    _prewarmed(image, **record)

    server, _ = _run(image, monkeypatch, "otherwise", _arm(image), cache=cache)

    assert server.found["module_gemm"] is False and server.held["module_gemm.so"] == (False, BUILT)


def test_a_farm_with_no_cache_leaves_out_its_link_to_a_module_built_once_under_another_value(image) -> None:
    _prewarmed(image)

    _, _, run = image.run(
        "farm-prewarmed", _arm(image, envs={"CU_NUM": "304"}, roots=_aiter_package_root(image)), cache=False
    )

    farm_jit = next(run.rglob("farm/site/aiter/jit"))
    assert (farm_jit / "module_other.so").is_symlink()
    assert not os.path.lexists(farm_jit / "module_gemm.so"), "built under the test's CU_NUM, not linked"
