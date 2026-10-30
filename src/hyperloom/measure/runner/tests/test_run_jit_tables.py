# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test that delivers a tuned table, through the whole runner (the shapes of ``test_run_jit``).

The image carries two prebuilt modules: ``module_gemm``, generated from the table ``AITER_CONFIG_GEMM_A8W8`` resolves to
(its command passes ``{AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE}``, as aiter's GEMM modules do), and ``module_other``,
generated from none. A test whose table differs from the image's, by the variable, by a patch to ``configs`` or by the
table a kernel tune made, must not be given the image's ``module_gemm``: the server (the fake grid) builds it, from the
test's table, in the test's JIT directory, and the image keeps its own."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hyperloom.measure.framework.base import FRAMEWORK_BASE_FILE, framework_envs
from hyperloom.measure.runner.tests import test_run_jit
from hyperloom.measure.runner.tests.test_run import _blob
from hyperloom.measure.runner.tests.test_run_jit import (  # noqa: F401 - fixtures
    CORE,
    TABLE_VARIABLE,
    Rig,
    _arm,
    _kind,
    env,
    key,
    rig,
)

IMAGE_TABLE = "M,N,K,kernelName\n1,2,3,gemm_k1\n"
TEST_TABLE = "M,N,K,kernelName\n7,8,9,gemm_k9\n"
DEFAULT = "configs/a8w8_tuned_gemm.csv"
IMAGE_MODULE = "image build"
INERT = "inert_patch:"
TOUCHED = "the patch or the tuned tables touch 1 aiter modules, built when first called (module_gemm)"
DECLARED = (
    'AITER_CONFIG_GEMM_A8W8 = os.getenv("AITER_CONFIG_GEMM_A8W8", f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv")\n'
    "class AITER_CONFIG:\n"
    "    @property\n"
    "    def AITER_CONFIG_GEMM_A8W8_FILE(self):\n"
    '        return self.get_config_file("AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm")\n'
)
GEMM_COMMAND = "f'{AITER_CSRC_DIR}/ck_gemm/gen_instances.py --working_path {{}} --tune_file {AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE}'"
REGISTRY_MISS = (
    "RuntimeError: gemm kernel 'gemm_k9' is not present in the compiled registry. The tuned CSV references a kernel "
    "that was not built into aiter.\n"
)


@pytest.fixture
def image(rig: Rig) -> Rig:
    """The rig's image with the two prebuilt modules, their config, the variable's declaration and its default table."""
    config = {
        "module_gemm": {"srcs": [], "extra_include": [], "blob_gen_cmd": GEMM_COMMAND},
        "module_other": {"srcs": [], "extra_include": [], "blob_gen_cmd": "''"},
    }
    files = {
        "aiter/jit/core.py": CORE + DECLARED,
        "aiter/jit/optCompilerConfig.json": json.dumps(config),
        "aiter/jit/module_gemm.so": IMAGE_MODULE,
        "aiter/jit/module_other.so": IMAGE_MODULE,
        f"aiter/{DEFAULT}": IMAGE_TABLE,
    }
    for relative, text in files.items():
        path = rig.bases / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return rig


def _table(run: Path, text: str) -> str:
    path = run / "tables" / "mine.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _by_variable(text: str):
    return lambda run: {"envs": {TABLE_VARIABLE: _table(run, text)}}


def _by_patch(rig: Rig, text: str):
    return lambda run: {"roots": [rig.root(run, "aiter", DEFAULT, text.encode())]}


def _by_kernel_tune(text: str):
    """What compile makes of a kernel tune's whole table: a file attachment the variable names. (Rows over the image's
    default table need the private tables of a local sandbox; a runner refuses them, and ``framework.tests.test_table_modules``
    resolves them.)"""

    def sheet(run: Path) -> dict:
        attachment = {"id": "k-0123456789abcdef", "kind": "file", "blob": _blob(run, text.encode())}
        placed = "${HL_ATTACH:k-0123456789abcdef}"
        return {"envs": {TABLE_VARIABLE: placed}, "attachments": [{**attachment, "manifest_sha256": "44" * 32}]}

    return sheet


def _deliver(rig: Rig, delivery):
    return lambda run: rig.sheet(run, **delivery(run))


def _measured(manifest) -> bool:
    """Whether nothing stale ran; the fake server never imports a patched package, which the runner calls inert."""
    outcome = manifest.outcome
    return outcome.failure_class is None or outcome.failure_reason.startswith(INERT)


def _assert_built_in_the_test(rig: Rig, manifest, status: dict) -> None:
    contributed = _kind(manifest, "aiter_jit").contribution
    assert _measured(manifest), "the module the table touches was built, so nothing stale ran"
    assert contributed is not None and contributed.files == 1, "module_gemm.so is the test's; module_other.so a link"
    assert TOUCHED in status["message"]
    assert (rig.bases / "aiter/jit/module_gemm.so").read_text() == IMAGE_MODULE, "the image keeps its module"
    assert (rig.bases / f"aiter/{DEFAULT}").read_text() == IMAGE_TABLE, "and its table"


@pytest.mark.parametrize(
    "delivery",
    [
        pytest.param(lambda rig: _by_variable(TEST_TABLE), id="variable"),
        pytest.param(lambda rig: _by_patch(rig, TEST_TABLE), id="configs-patch"),
        pytest.param(lambda rig: _by_kernel_tune(TEST_TABLE), id="kernel-tune"),
    ],
)
def test_a_table_that_differs_from_the_images_has_its_module_built_in_the_test_and_not_linked(image, delivery) -> None:
    manifest, status, _ = image.run("table", _deliver(image, delivery(image)))

    _assert_built_in_the_test(image, manifest, status)


@pytest.mark.parametrize(
    "delivery",
    [
        pytest.param(lambda rig: _by_variable(IMAGE_TABLE), id="variable"),
        pytest.param(lambda rig: _by_kernel_tune(IMAGE_TABLE), id="kernel-tune"),
    ],
)
def test_a_table_with_the_images_content_keeps_the_images_module(image, delivery) -> None:
    manifest, status, _ = image.run("same", _deliver(image, delivery(image)))

    contributed = _kind(manifest, "aiter_jit").contribution
    assert manifest.outcome.failure_class is None
    assert contributed is None or contributed.files == 0, "both modules are the image's links"
    assert "aiter modules, built when first called" not in status["message"]


def test_a_patch_to_another_package_file_keeps_the_images_module(image) -> None:
    manifest, status, _ = image.run(
        "package", _arm(image, roots=lambda run: [image.root(run, "aiter", "ops/mha.py", b"X = 2\n")])
    )

    assert _measured(manifest)
    assert "aiter modules, built when first called" not in status["message"]


def test_a_module_this_test_built_from_its_table_that_lacks_a_kernel_is_the_candidates(image, monkeypatch) -> None:
    monkeypatch.setattr(test_run_jit, "BUILD_LOG", test_run_jit.BUILD_LOG + REGISTRY_MISS)

    manifest, _, _ = image.run("lacks", _deliver(image, _by_variable(TEST_TABLE)), fail=True)

    assert manifest.outcome.failure_class == "candidate"
    assert manifest.outcome.failure_reason.startswith("kernel_not_built: module_gemm was built by this test")


def test_a_configs_patch_reaches_the_server_of_a_sandbox_that_reads_private_copies_of_the_tables(
    image, monkeypatch
) -> None:
    """A local sandbox's runner gives the server private copies of the session's tables and names them by every variable;
    the copies carry the patched table, so the module the patch makes stale is built from it and not the image's."""
    for name, value in framework_envs(image.tmp_path / "framework", str(image.bases / "aiter" / "configs")).items():
        monkeypatch.setenv(name, value)
    served: list[str] = []
    grid = test_run_jit._CompilingGrid.__call__

    async def reading(self, **kwargs):
        served.append(Path(os.environ[TABLE_VARIABLE]).read_text(encoding="utf-8"))
        return await grid(self, **kwargs)

    monkeypatch.setattr(test_run_jit._CompilingGrid, "__call__", reading)

    manifest, status, run = image.run("private", _deliver(image, _by_patch(image, TEST_TABLE)))

    assert served and set(served) == {TEST_TABLE}
    _assert_built_in_the_test(image, manifest, status)
    document = json.loads((run / "out" / "files" / FRAMEWORK_BASE_FILE).read_text())
    assert document["isolation"] == "private_copies" and document["configs_patched"] == ["a8w8_tuned_gemm.csv"]
