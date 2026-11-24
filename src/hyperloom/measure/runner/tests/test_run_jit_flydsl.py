# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlyDSL's compiled launches through the runner, the local backend and the host.

The stand-in server compiles one FlyDSL launch into the directory ``FLYDSL_RUNTIME_CACHE_DIR`` names (the grid of
:mod:`test_run_jit`), and here it also notes what it found there. So "a patched test never publishes" is a whole runner
whose signed result stages nothing for the kind, and "the image's prebuilt launches stay visible" is what the server saw.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.measure.runner.tests import test_run_jit as base
from hyperloom.measure.runner.tests.test_run_jit import FLYDSL_LAUNCH, Rig, _arm, _keys, _kind, _states
from hyperloom.measure.runner.tests.test_run import env, key  # noqa: F401 - fixtures

VARIABLE = "FLYDSL_RUNTIME_CACHE_DIR"
PREBUILT_LAUNCH = f"gemm_a16w16_gfx950_{'1' * 32}/{'2' * 16}.pkl"
COLD_LINE = "monokernel kimi_k31_hip on gfx950: widths [1, 2, 4] built in 1071.2 s\n"
WARM_LINE = "monokernel kimi_k31_hip on gfx950: widths [1, 2, 4] built in 1.4 s\n"
#: Files of the sources a FlyDSL launch is built from, below the package each is imported as.
COVERED = [
    ("flydsl", "compiler/jit_function.py"),
    ("sgl_amd_mono", "kimi_k31/gfx950/chain.py"),
    ("aiter", "ops/flydsl/gemm.py"),
    ("sglang", "srt/kernel_plugins/amd/binding.py"),
    ("sglang", "srt/monokernels/kimi.py"),
    ("sglang", "srt/models/kimi_k3.py"),
]


class _SeeingGrid(base._CompilingGrid):
    """The compiling grid, noting where FlyDSL was pointed and what it found there, and saying how long the monokernel
    build took: long when it compiled the launch, a moment when it found it."""

    seen: list[dict] = []

    async def __call__(self, **kwargs):
        if not kwargs["launch"].server_already_ready:
            where = os.environ.get(VARIABLE)
            found = {}
            if where:
                for path in sorted(Path(where).rglob("*")):
                    if path.is_file():
                        found[path.relative_to(where).as_posix()] = "link" if path.is_symlink() else "file"
            log = self.files / "variant_0" / "server.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as out:
                out.write(WARM_LINE if FLYDSL_LAUNCH in found else COLD_LINE)
            _SeeingGrid.seen.append({"where": where, "found": found})
        return await super().__call__(**kwargs)


@pytest.fixture
def rig(tmp_path, env, key, monkeypatch) -> Rig:
    monkeypatch.setattr(base, "_CompilingGrid", _SeeingGrid)
    monkeypatch.setattr(_SeeingGrid, "seen", [])
    made = Rig(tmp_path, env, key, monkeypatch)
    for package, path in [*COVERED, ("sgl_amd_mono", "__init__.py"), ("flydsl", "__init__.py")]:
        target = made.bases / package / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("x = 1\n", encoding="utf-8")
    return made


def _patch(rig: Rig, package: str, path: str, data: bytes = b"x = 2\n"):
    return lambda run: [rig.root(run, package, path, data)]


def test_the_server_compiles_into_the_tests_own_directory_and_the_next_test_loads_what_it_compiled(rig) -> None:
    first, _, first_run = rig.run("first", _arm(rig))
    rig.publish(first_run, first)
    second, _, _ = rig.run("second", _arm(rig))

    cold, warm = _SeeingGrid.seen
    assert cold["where"] is not None and Path(cold["where"]).is_relative_to(rig.host_tmp), (
        "the test's compile directory"
    )
    assert cold["found"] == {} and warm["found"] == {FLYDSL_LAUNCH: "file"}
    assert cold["where"] != warm["where"], "no two tests share a writable directory"
    assert (_states(first)["flydsl"], _states(second)["flydsl"]) == ("miss", "hit")
    assert _kind(first, "flydsl").contribution is not None and _kind(second, "flydsl").contribution is None
    assert (first.timing.jit_compile_sec, second.timing.jit_compile_sec) == (12.5 + 1071.2, 1.4), "the hit shows"
    assert (first.jit_cache.builds_seen, second.jit_cache.builds_seen) == (1, 0)


@pytest.mark.parametrize(("package", "path"), COVERED)
def test_a_patch_to_a_source_of_a_launch_gets_a_key_of_its_own_and_never_publishes(rig, package, path) -> None:
    base_run_manifest, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, base_run_manifest)

    one, status, one_run = rig.run("one", _arm(rig, roots=_patch(rig, package, path)))
    two, _, _ = rig.run("two", _arm(rig, roots=_patch(rig, package, path, b"x = 3\n")))

    assert len({_keys(base_run_manifest)["flydsl"], _keys(one)["flydsl"], _keys(two)["flydsl"]}) == 3
    assert _states(one)["flydsl"] == "miss", "the baseline's launches are not given to it"
    assert _kind(one, "flydsl").contribution is None and not (one_run / "out" / "jit-cache" / "flydsl").exists()
    assert "jit cache: flydsl is read only in this test" in status["message"]
    assert rig.publish(one_run, one) == (2 if package == "aiter" else 0), (
        "the aiter key covers an aiter patch, no other"
    )
    assert not (rig.store.root / "snapshots" / "flydsl" / _keys(one)["flydsl"]).exists()


def test_a_patch_elsewhere_is_given_the_baselines_launches_and_adds_none(rig) -> None:
    baseline, _, base_run = rig.run("base", _arm(rig))
    rig.publish(base_run, baseline)

    patched, _, patched_run = rig.run("patched", _arm(rig, roots=_patch(rig, "sglang", "srt/a.py")))

    assert _keys(patched)["flydsl"] == _keys(baseline)["flydsl"] and _states(patched)["flydsl"] == "hit"
    assert _SeeingGrid.seen[-1]["found"] == {FLYDSL_LAUNCH: "file"}
    assert _kind(patched, "flydsl").contribution is None and rig.publish(patched_run, patched) == 0


@pytest.mark.parametrize(
    ("envs", "splits"),
    [
        ({"SGLANG_AMD_MONO_K31_WIDE": "0"}, True),
        ({"FLYDSL_COMPILE_OPT_LEVEL": "2"}, True),
        ({"ARCH": "gfx942"}, True),
        ({"SGLANG_AMD_MONO_CACHE_DIR": "/elsewhere"}, False),
        ({"SGLANG_X": "1"}, False),
    ],
)
def test_a_flydsl_or_plugin_variable_splits_the_key_and_a_location_or_another_variable_does_not(
    rig, envs, splits
) -> None:
    baseline, _, _ = rig.run("base", _arm(rig, envs={"SGLANG_AMD_MONO_K31_WIDE": "1"}))
    other, _, _ = rig.run("other", _arm(rig, envs={"SGLANG_AMD_MONO_K31_WIDE": "1", **envs}))

    assert (_keys(other)["flydsl"] != _keys(baseline)["flydsl"]) is splits
    assert _keys(other)["triton"] == _keys(baseline)["triton"]


def test_the_images_prebuilt_launches_are_visible_to_the_server_and_are_not_contributed(rig) -> None:
    image = rig.bases / "aiter" / "jit" / "flydsl_cache"
    (image / PREBUILT_LAUNCH).parent.mkdir(parents=True)
    (image / PREBUILT_LAUNCH).write_bytes(b"prebuilt")

    manifest, _, run = rig.run("first", _arm(rig))

    assert _SeeingGrid.seen[0]["found"] == {PREBUILT_LAUNCH: "link"}
    assert _kind(manifest, "flydsl").contribution.files == 1, "only the launch the server compiled"
    assert sorted(p.relative_to(image).as_posix() for p in image.rglob("*.pkl")) == [PREBUILT_LAUNCH]
    rig.publish(run, manifest)
    rig.run("second", _arm(rig))
    assert _SeeingGrid.seen[1]["found"] == {PREBUILT_LAUNCH: "link", FLYDSL_LAUNCH: "file"}


def test_a_sheet_that_sets_flydsls_cache_directory_itself_keeps_it_and_the_kind_is_not_used(rig) -> None:
    manifest, status, _ = rig.run("own", _arm(rig, envs={VARIABLE: "/somewhere/else"}))

    assert _states(manifest)["flydsl"] == "disabled" and _states(manifest)["triton"] == "miss"
    assert rig.grid.configs[0]["benchmark"]["envs"][VARIABLE] == "/somewhere/else"
    assert _SeeingGrid.seen[0]["where"] is None, "the runner names no directory of its own for it"
    assert f"jit cache: flydsl is not used: the sheet sets {VARIABLE} itself" in status["message"]
