# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An executable root laid by the farm: its build's executables copied out of the store, proven to be the build the sheet
names, and found first on the server's ``PATH`` or refused."""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hyperloom.contracts.recipe import ComponentPatches
from hyperloom.measure.farm import FarmError, build_farm, built_roots, place_farm, verify_farm
from hyperloom.measure.jit.native.laying import NamedBuild, SheetBuilds, manifest_sha256
from hyperloom.measure.jit.native.record import BuildRecord
from hyperloom.measure.runner.native.builder import BuildMachine, BuildRequest, plan
from hyperloom.measure.runner.native.recipes import NativeRecipe
from hyperloom.measure.runner.native.service import ensure_build
from hyperloom.measure.runner.native.tests._native import (
    BASE_SOURCE,
    BUILDER_IMAGE,
    COMPONENT,
    Bench,
    OUTPUT,
    CountingPreflight,
    compile_tree,
    compiled,
    executable_recipe,
    fake_recipe,
)
from hyperloom.measure.spec import EnvRoot
from hyperloom.measure.tests._recipes import COMMIT, IMAGE


def _build(bench: Bench, recipe: NativeRecipe) -> BuildRecord:
    stack = ComponentPatches(COMPONENT, "https://example.invalid/fake.git", COMMIT, ())
    request = BuildRequest(recipe, stack, bench.upstream, bench.image, BuildMachine(IMAGE, BUILDER_IMAGE, "gfx942"))
    inputs, root = plan(request, bench.blobs)
    ensured = ensure_build(
        bench.store, inputs, lambda scratch: compile_tree(request, root, bench.blobs, scratch), bench.signer
    )
    return ensured.record


def _root(root_id: str = COMPONENT) -> EnvRoot:
    return EnvRoot(root_id, "executable", COMPONENT, COMPONENT, COMMIT, ())


def _builds(bench: Bench, record: BuildRecord) -> SheetBuilds:
    return SheetBuilds(bench.store, {COMPONENT: NamedBuild(COMPONENT, record.key, manifest_sha256(record))}, IMAGE)


def _no_blob(address: str) -> bytes:
    raise AssertionError(f"an executable root reads no blob: {address}")


def test_an_executable_root_lays_its_builds_executables_and_the_servers_path_must_find_them_first(
    tmp_path: Path,
) -> None:
    bench = Bench(tmp_path)
    record = _build(bench, executable_recipe())
    roots = built_roots([_root()], _builds(bench, record))

    farm = build_farm(roots, tmp_path / "farm", python=sys.executable, read_blob=_no_blob, inherited={})

    laid = tmp_path / "farm" / COMPONENT / OUTPUT
    assert farm is not None
    assert (farm.executables, farm.bin, farm.packages, farm.imports) == ((laid,), (laid.parent,), (), False)
    assert laid.read_bytes() == compiled(BASE_SOURCE) and os.stat(laid).st_mode & stat.S_IXUSR
    assert [root.kind for root in farm.roots] == ["executable"]
    verify_farm(farm, {"PATH": f"{laid.parent}{os.pathsep}/usr/bin"}, python=sys.executable, cwd=tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / laid.name).write_bytes(b"#!/bin/sh\n")
    (shadow / laid.name).chmod(0o755)
    for path in (f"{shadow}{os.pathsep}{laid.parent}", "/usr/bin"):
        with pytest.raises(FarmError, match=f"{laid.name} would be run from"):
            verify_farm(farm, {"PATH": path}, python=sys.executable, cwd=tmp_path)


def test_a_build_whose_outputs_are_not_executables_is_refused_for_an_executable_root(tmp_path: Path) -> None:
    bench = Bench(tmp_path)
    record = _build(bench, replace(fake_recipe(CountingPreflight()), against_image=False))

    with pytest.raises(FarmError, match="has outputs that are not executables"):
        built_roots([_root()], _builds(bench, record))


def test_an_executable_root_without_its_build_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FarmError, match="its build needs the native build store"):
        built_roots([_root()], None)
    with pytest.raises(FarmError, match="an executable root needs its build"):
        build_farm([_root()], tmp_path / "farm", python=sys.executable, read_blob=_no_blob, inherited={})


def _farm(tmp_path: Path):
    bench = Bench(tmp_path)
    roots = built_roots([_root()], _builds(bench, _build(bench, executable_recipe())))
    return build_farm(roots, tmp_path / "farm", python=sys.executable, read_blob=_no_blob, inherited={})


def _point(tmp_path: Path, envs: dict[str, str]) -> Path:
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"framework": "llamacpp", "model": "m", "envs": envs}}))
    return point


@pytest.mark.parametrize(
    ("named", "inherited", "after"),
    [
        ({"PATH": "/yaml/bin:/opt/rocm/bin"}, {"PATH": "/inherited/bin"}, "/yaml/bin:/opt/rocm/bin"),
        ({}, {"PATH": "/inherited/bin:/opt/rocm/bin"}, "/inherited/bin:/opt/rocm/bin"),
        ({}, {}, os.defpath),
    ],
    ids=["the-yaml-path", "the-inherited-path", "the-default-path"],
)
def test_the_farm_puts_its_executables_first_on_the_path_the_server_would_get_and_keeps_that_path_whole(
    tmp_path: Path, named: dict[str, str], inherited: dict[str, str], after: str
) -> None:
    farm = _farm(tmp_path)
    point = _point(tmp_path, named)

    envs = place_farm(point, farm, inherited)

    assert envs["PATH"] == os.pathsep.join([str(farm.bin[0]), *after.split(os.pathsep)])
    assert yaml.safe_load(point.read_text())["benchmark"]["envs"] == dict(envs)
    assert "PYTHONPATH" not in envs, "a farm of executables alone puts nothing on the import path"
    verify_farm(farm, envs, python=sys.executable, cwd=tmp_path)
