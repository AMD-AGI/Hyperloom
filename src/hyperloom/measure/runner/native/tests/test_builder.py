# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stack of the stand-in native component built end to end into the store: key, compile, harvest, publish, read."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.compile import PatchApplyError
from hyperloom.measure.runner.native.builder import plan
from hyperloom.measure.jit.native.record import FAILED, LOG_NAME, READY, RECORD_FILES
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native.tests._native import KERNEL, OUTPUT, Bench, compiled

PATCHED = "int kernel(void) { return 2; }\n"


@pytest.fixture
def bench(tmp_path: Path) -> Bench:
    return Bench(tmp_path)


def _build(bench: Bench, *patches):
    return bench.build(*patches)


def test_patched_build_publishes_the_output_that_differs_from_the_image(bench: Bench, tmp_path: Path) -> None:
    ensured = _build(bench, bench.patch({KERNEL: PATCHED}))

    record = ensured.record
    assert (ensured.reused, record.status, [o.path for o in record.outputs]) == (False, READY, [OUTPUT])
    entry = bench.store.entry_dir(record.inputs)
    assert sorted(p.name for p in entry.iterdir()) == sorted([*RECORD_FILES[READY], LOG_NAME])
    bench.store.fetch(record, tmp_path / "farm")
    assert (tmp_path / "farm" / OUTPUT).read_bytes() == compiled(PATCHED)
    assert record.toolchain == {"compiler": "stand-in", "gfx": "gfx942", "max_jobs": "1"}


def test_a_second_request_of_the_key_reuses_the_entry_without_compiling(bench: Bench) -> None:
    patch = bench.patch({KERNEL: PATCHED})
    first, second = _build(bench, patch), _build(bench, patch)

    assert (first.reused, second.reused, len(bench.preflight.calls)) == (False, True, 1)
    assert second.record == first.record


def test_stacks_that_yield_the_same_sources_share_one_key(bench: Bench) -> None:
    direct = bench.patch({KERNEL: PATCHED})
    detour = bench.patch({KERNEL: "int kernel(void) { return 3; }\n"})
    back = bench.patch({KERNEL: PATCHED}, before={KERNEL: "int kernel(void) { return 3; }\n"})

    key_direct = plan(bench.request(direct), bench.blobs)[0].key
    assert plan(bench.request(detour, back), bench.blobs)[0].key == key_direct
    assert plan(bench.request(detour), bench.blobs)[0].key != key_direct


def test_unpatched_build_equal_to_the_image_lists_no_output(bench: Bench) -> None:
    record = _build(bench).record

    assert (record.status, record.outputs) == (READY, ())


def test_a_compile_error_is_stored_signed_and_answered_again_without_a_rebuild(bench: Bench) -> None:
    patch = bench.patch({KERNEL: "#error\n"})
    first, second = _build(bench, patch), _build(bench, patch)

    assert (first.record.status, second.reused, len(bench.preflight.calls)) == (FAILED, True, 1)
    assert first.record.failure == "compile_error: the compile exited 1"
    entry = bench.store.entry_dir(first.record.inputs)
    assert sorted(p.name for p in entry.iterdir()) == sorted([*RECORD_FILES[FAILED], LOG_NAME])
    assert "error: stop" in (entry / LOG_NAME).read_text()


@pytest.mark.parametrize(
    ("marker", "reason"),
    [
        ("#symlink", "an output must be a regular file reached without a symlink (symlink or not a directory"),
        ("#notelf", "an output must be an ELF file"),
        ("#big", "is more than the 1024 an output may be"),
        ("#none", "the build made none of its declared outputs"),
    ],
)
def test_outputs_not_as_declared_are_the_candidates_compile_error(bench: Bench, marker: str, reason: str) -> None:
    record = _build(bench, bench.patch({KERNEL: f"{marker}\n"})).record

    assert record.status == FAILED
    assert record.failure.startswith("compile_error: ") and reason in record.failure


def test_a_failed_preflight_is_infra_and_stores_nothing(bench: Bench) -> None:
    patch = bench.patch({KERNEL: PATCHED})
    bench.preflight.problem = "preflight_disk: 1 GB free"
    with pytest.raises(BuildInfraError, match="preflight_disk"):
        _build(bench, patch)
    inputs = plan(bench.request(patch), bench.blobs)[0]
    assert not bench.store.entry_dir(inputs).exists()

    bench.preflight.problem = None
    assert _build(bench, patch).reused is False


def test_a_compiler_killed_by_a_signal_is_infra_and_stores_nothing(bench: Bench) -> None:
    patch = bench.patch({KERNEL: "#signal\n"})
    with pytest.raises(BuildInfraError, match="timed out or was killed"):
        _build(bench, patch)
    assert not bench.store.entry_dir(plan(bench.request(patch), bench.blobs)[0]).exists()


def test_a_stack_that_changes_the_build_definition_is_refused(bench: Bench) -> None:
    patch = bench.patch(
        {"native/CMakeLists.txt": "project(evil)\n"}, before={"native/CMakeLists.txt": "project(fake)\n"}
    )
    with pytest.raises(
        PatchApplyError, match="'native/CMakeLists.txt' is a native or binary file \\(CMakeLists.txt\\)"
    ):
        plan(bench.request(patch), bench.blobs)


def test_the_compile_sees_no_variable_of_the_service(bench: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HL_TEST_SERVICE_SECRET", "x")
    record = _build(bench, bench.patch({KERNEL: PATCHED})).record

    names = (bench.store.entry_dir(record.inputs) / LOG_NAME).read_text().split()
    assert "HL_TEST_SERVICE_SECRET" not in names
    assert {"AMDGPU_TARGET", "MAX_JOBS", "PATH"} <= set(names)
