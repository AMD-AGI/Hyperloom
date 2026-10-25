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
import os
from pathlib import Path

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
    _by_variable,
    _deliver,
    image,
)
from hyperloom.measure.socket import TestCall

BUILT = b"\x7fELF built by the test"
WANTED = ("module_gemm", "module_other", "module_new")


class _JitServer:
    """What the server's aiter does with its JIT directory; what it found and where it built, by test."""

    def __init__(self, rig: Rig, monkeypatch) -> None:
        self.image_jit = rig.bases / "aiter" / "jit"
        self.found: dict[str, bool] = {}
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
    return server, run


def test_a_test_with_no_patch_and_no_cache_builds_in_its_own_jit_directory_beside_the_images_modules(
    image, monkeypatch
) -> None:
    server, run = _run(image, monkeypatch, "plain", _arm(image), cache=False)

    own = run / "scratch" / "test" / OWN_AITER_JIT_DIR
    assert server.jit_dir == own
    assert server.found == {"module_gemm": True, "module_other": True, "module_new": False}
    assert (own / "module_gemm.so").is_symlink() and (own / "module_gemm.so").read_text() == IMAGE_MODULE
    assert (own / "module_new.so").read_bytes() == BUILT and not (own / "module_new.so").is_symlink()


def test_a_module_the_tests_table_makes_stale_is_not_linked_but_built(image, monkeypatch) -> None:
    server, run = _run(image, monkeypatch, "table", _deliver(image, _by_variable(TEST_TABLE)), cache=False)

    own = run / "scratch" / "test" / OWN_AITER_JIT_DIR
    assert server.found == {"module_gemm": False, "module_other": True, "module_new": False}
    assert (own / "module_gemm.so").read_bytes() == BUILT


def test_a_jit_directory_the_sheet_names_is_left_as_it_is(image, monkeypatch, tmp_path) -> None:
    named = tmp_path / "named_jit"
    named.mkdir()
    server, run = _run(image, monkeypatch, "named", _arm(image, process_env={"AITER_JIT_DIR": str(named)}), cache=False)

    assert server.jit_dir == named and not (run / "scratch" / "test" / OWN_AITER_JIT_DIR).exists()


def test_a_test_the_cache_gives_a_jit_directory_gets_no_second_one(image, monkeypatch) -> None:
    server, run = _run(image, monkeypatch, "cached", _arm(image), cache=True)

    assert server.jit_dir == run / "scratch" / "test" / "jit" / "aiter_jit", "the cache's directory"
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
