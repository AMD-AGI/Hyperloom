# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test that patches aiter's ``csrc`` or ``hsa``, through the whole runner (the shapes of ``test_run_jit``).

The image carries two prebuilt modules, ``module_gemm`` built from ``csrc/kernels/k.cu`` and ``module_other`` from another
file. The sheet's second root, ``aiter.meta``, patches ``k.cu``: the test's JIT directory must not link the image's
``module_gemm``, so the server (the fake grid) builds it, and ``module_other`` stays the image's. What the result and the
cache then say is what is checked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.runner.tests import test_run_jit
from hyperloom.measure.runner.tests.test_run_jit import (  # noqa: F401 - fixtures
    Rig,
    _arm,
    _keys,
    _kind,
    _states,
    env,
    key,
    rig,
)

PATCHED_K = b"// patched k.cu\n"
CSRC = "f'{AITER_CSRC_DIR}/kernels/"


def _entry(source: str) -> dict:
    return {"srcs": [f"{CSRC}{source}'"], "extra_include": [], "blob_gen_cmd": "''"}


@pytest.fixture
def image(rig: Rig) -> Rig:
    """The rig's image with the two prebuilt modules, their config and their sources."""
    config = {"module_gemm": _entry("k.cu"), "module_other": _entry("other.cu")}
    files = {
        "aiter/jit/optCompilerConfig.json": json.dumps(config),
        "aiter/jit/module_gemm.so": "image build",
        "aiter/jit/module_other.so": "image build",
        "csrc/kernels/k.cu": "// image k.cu\n",
        "csrc/kernels/other.cu": "// other.cu\n",
    }
    for relative, text in files.items():
        path = rig.bases / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return rig


def _meta_patch(rig: Rig, path: str = "csrc/kernels/k.cu", data: bytes = PATCHED_K):
    def roots(run: Path) -> list:
        root = rig.root(run, "aiter_meta", path, data)
        return [{**root, "id": "aiter.meta"}]

    return roots


def test_the_server_builds_the_module_the_patch_touches_and_keeps_the_one_it_does_not(image) -> None:
    manifest, status, _ = image.run("patched", _arm(image, roots=_meta_patch(image)))

    contributed = _kind(manifest, "aiter_jit").contribution
    assert manifest.outcome.failure_class is None, "the touched module was built, so nothing stale ran"
    assert contributed is not None and contributed.files == 1, (
        "module_gemm.so is new; module_other.so is the image's link"
    )
    assert "the patch touches 1 aiter modules, built when first called (module_gemm)" in status["message"]


def test_a_second_test_of_the_same_patch_finds_the_module_the_first_built(image) -> None:
    first, _, first_run = image.run("first", _arm(image, roots=_meta_patch(image)))
    image.publish(first_run, first)

    second, _, _ = image.run("second", _arm(image, roots=_meta_patch(image)))

    assert _states(second)["aiter_jit"] == "hit" and second.jit_cache.builds_seen == 0
    assert second.outcome.failure_class is None, "the module came from the cache, so it is the patch's"


def test_the_patched_meta_tree_is_part_of_the_aiter_key_and_another_patch_is_another_key(image) -> None:
    base, _, _ = image.run("base", _arm(image))
    patched, _, _ = image.run("patched", _arm(image, roots=_meta_patch(image)))
    other, _, _ = image.run("other", _arm(image, roots=_meta_patch(image, data=b"// another\n")))

    keys = {name: _keys(manifest)["aiter_jit"] for name, manifest in (("base", base), ("p", patched), ("o", other))}
    assert len(set(keys.values())) == 3
    assert _keys(patched)["triton"] == _keys(base)["triton"]


def test_the_key_of_a_test_with_no_meta_root_is_the_one_it_was(image) -> None:
    base, _, _ = image.run("base", _arm(image))
    package_only, _, _ = image.run(
        "package", _arm(image, roots=lambda run: [image.root(run, "aiter", "ops/mha.py", b"X = 1\n")])
    )

    assert _keys(base)["aiter_jit"] != _keys(package_only)["aiter_jit"]
    assert _keys(image.run("again", _arm(image))[0])["aiter_jit"] == _keys(base)["aiter_jit"]


def test_a_runner_without_a_cache_refuses_a_csrc_patch_instead_of_running_the_images_modules(image) -> None:
    status = image.refused("nocache", _arm(image, roots=_meta_patch(image)), cache=False)

    assert status["error_class"] == "preflight_mismatch"
    assert "HYPERLOOM_JIT_CACHE is not set" in status["message"]


def test_a_sheet_that_sets_the_aiter_jit_directory_itself_is_refused_for_a_csrc_patch(image) -> None:
    status = image.refused("own", _arm(image, roots=_meta_patch(image), envs={"AITER_JIT_DIR": "/somewhere"}))

    assert status["error_class"] == "preflight_mismatch"
    assert "the sheet sets AITER_JIT_DIR itself" in status["message"]


def test_a_csrc_patch_with_aiter_rebuild_is_allowed_because_every_module_is_built(image) -> None:
    manifest, _, _ = image.run("rebuild", _arm(image, roots=_meta_patch(image), envs={"AITER_REBUILD": "1"}))

    assert _states(manifest)["aiter_jit"] == "disabled" and manifest.outcome.failure_class is None


def test_a_server_that_called_the_module_from_the_images_copy_is_a_failed_measurement_and_not_a_result(
    image, monkeypatch
) -> None:
    """A copy of the image's module that is in the test's directory when the patch touched it is stale code."""
    from hyperloom.measure.jit import session as session_module

    real = session_module._link_prebuilt
    monkeypatch.setattr(
        session_module, "_link_prebuilt", lambda prebuilt, content, skip: real(prebuilt, content, frozenset())
    )

    manifest, _, _ = image.run("stale", _arm(image, roots=_meta_patch(image)))

    assert manifest.outcome.failure_class == "infra"
    assert manifest.outcome.failure_reason.startswith("stale_aiter_module: module_gemm ran as the image built it")


def _failed_build(module: str) -> str:
    """What aiter writes when the build of ``module`` fails: the record, then the error that stops the server."""
    return (
        f"[aiter] start build [{module}] under /j/build/{module}\n"
        f"failed jit build [{module}]\u2193\u2193\n-->[History]: Traceback\n"
        "/sandbox/farm/site/aiter_meta/csrc/kernels/k.cu:12:5: error: use of undeclared identifier 'x'\n"
        f"failed jit build [{module}]\u2191\u2191\nRuntimeError: [aiter] build [{module}] failed !!!!!!\n"
    )


def test_a_server_that_failed_while_building_a_touched_module_is_the_candidates_compile_error(
    image, monkeypatch
) -> None:
    monkeypatch.setattr(test_run_jit, "BUILD_LOG", _failed_build("module_gemm"))

    manifest, _, _ = image.run("broken", _arm(image, roots=_meta_patch(image)), fail=True)

    assert manifest.outcome.failure_class == "candidate"
    assert manifest.outcome.failure_reason == (
        "compile_error: module_gemm: kernels/k.cu:12: use of undeclared identifier 'x'"
    )


def test_the_same_failure_in_a_module_the_patch_did_not_touch_is_not_the_candidates(image, monkeypatch) -> None:
    monkeypatch.setattr(test_run_jit, "BUILD_LOG", _failed_build("module_other"))

    manifest, _, _ = image.run("other", _arm(image, roots=_meta_patch(image)), fail=True)

    assert not (manifest.outcome.failure_reason or "").startswith("compile_error")
