# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch to aiter's ``csrc`` or ``hsa`` is laid over the meta directory of the image: a second root, ``aiter.meta``, beside
the package root. The image is a develop install (the checkout holds aiter/, csrc/, hsa/ and the ``install_mode`` file says
develop), as the measured one is. Acceptance refuses what cannot be measured by name, and a stack that touches too many of
aiter's prebuilt modules, whose first build would not fit the time it is given."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import (
    LocalBlobStore,
    PackageBase,
    PatchApplyError,
    TooManyModulesError,
    compile_environment,
)
from hyperloom.measure.jit.modules import ESTIMATED_MODULE_SECONDS, MAX_REBUILD_SECONDS
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with, write_tree

OLD, NEW = "// old\n", "// new\n"
CSRC = "f'{AITER_CSRC_DIR}/"
LIMIT_MODULES = MAX_REBUILD_SECONDS // ESTIMATED_MODULE_SECONDS


def module_entry(index: int) -> dict:
    return {"srcs": [f"{CSRC}kernels/m{index}.cu'"], "extra_include": [], "blob_gen_cmd": "''"}


def aiter_image(root: Path, modules: int = 3, *, config: bool = True) -> PackageBase:
    """A develop-install checkout: the package, its prebuilt modules and config, and the sources beside it."""
    files = {
        "aiter/__init__.py": "",
        "aiter/ops/mha.py": OLD,
        "aiter/install_mode": "develop",
        "csrc/include/common.h": OLD,
        "hsa/gfx942/pa/pa_bf16.csv": "a,b\n",
        "hsa/gfx942/pa/pa_bf16.co": "binary",
        "hsa/codegen.py": "# codegen\n",
    }
    for index in range(modules):
        files[f"csrc/kernels/m{index}.cu"] = '#include "common.h"\n'
        files[f"aiter/jit/module_m{index}.so"] = "binary"
    if config:
        files["aiter/jit/optCompilerConfig.json"] = json.dumps(
            {f"module_m{i}": module_entry(i) for i in range(modules)}
        )
    write_tree(root, files)
    return PackageBase("aiter", root / "aiter", "amd-aiter", "0.1.23")


def compile_aiter_environment(tmp_path: Path, base: PackageBase, files: dict[str, str]):
    """The compiled environment of a one-patch stack of aiter that changes ``files`` (upstream path -> new text)."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    root = base.path.parent
    before = {path: (root / path).read_text() if (root / path).is_file() else OLD for path in files}
    patch = git_patch(tmp_path / "up", before, files)
    recipe = recipe_with((put_patch(blobs, "p.patch", patch),), component="aiter")
    return compile_environment(
        recipe, RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=lambda layout: base
    )


def compile_aiter(tmp_path: Path, base: PackageBase, files: dict[str, str]):
    environment = compile_aiter_environment(tmp_path, base, files)
    return environment.roots, LocalBlobStore(environment.blob_root)


def test_a_patch_over_the_package_and_csrc_is_two_roots_and_the_image_stays_as_it_was(tmp_path):
    base = aiter_image(tmp_path / "image")

    (package, meta), blobs = compile_aiter(
        tmp_path, base, {"aiter/ops/mha.py": NEW, "csrc/kernels/m0.cu": '#include "common.h"\n// patched\n'}
    )

    assert (package.id, package.import_name, [e.path for e in package.entries]) == ("aiter", "aiter", ["ops/mha.py"])
    assert (meta.id, meta.import_name, meta.kind) == ("aiter.meta", "aiter_meta", "package")
    (entry,) = meta.entries
    assert (entry.path, entry.op) == ("csrc/kernels/m0.cu", "upsert")
    assert blobs.get(entry.blob).endswith(b"// patched\n")
    assert (base.path.parent / "csrc/kernels/m0.cu").read_text() == '#include "common.h"\n'


def test_a_patch_that_changes_only_csrc_leaves_the_package_root_empty(tmp_path):
    base = aiter_image(tmp_path / "image")

    (package, meta), _ = compile_aiter(tmp_path, base, {"csrc/kernels/m1.cu": NEW})

    assert (package.entries, [e.path for e in meta.entries]) == ((), ["csrc/kernels/m1.cu"])


def test_an_hsa_table_is_laid_in_the_meta_directory(tmp_path):
    base = aiter_image(tmp_path / "image")

    (_, meta), _ = compile_aiter(tmp_path, base, {"hsa/gfx942/pa/pa_bf16.csv": "a,b\n1,2\n"})

    assert [e.path for e in meta.entries] == ["hsa/gfx942/pa/pa_bf16.csv"]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("hsa/gfx942/pa/pa_bf16.co", "is a native or binary file (*.co)"),
        ("3rdparty/composable_kernel/include/ck/ck.hpp", "is outside the installed package"),
    ],
)
def test_a_code_object_and_a_tree_that_is_no_section_are_refused_naming_the_path(tmp_path, path, message):
    base = aiter_image(tmp_path / "image")

    with pytest.raises(PatchApplyError) as refused:
        compile_aiter(tmp_path, base, {path: NEW})

    assert f"{path!r} {message}" in str(refused.value)


def test_a_header_that_every_module_includes_is_refused_with_the_count_and_the_seconds(tmp_path):
    modules = LIMIT_MODULES + 1
    base = aiter_image(tmp_path / "image", modules)

    with pytest.raises(TooManyModulesError) as refused:
        compile_aiter(tmp_path, base, {"csrc/include/common.h": NEW})

    message = str(refused.value)
    assert f"touches {modules} of aiter's prebuilt modules" in message
    assert f"about {modules * ESTIMATED_MODULE_SECONDS} s" in message
    assert f"more than {MAX_REBUILD_SECONDS} s is refused" in message


def test_the_same_header_is_accepted_when_the_modules_it_touches_fit_the_time(tmp_path):
    base = aiter_image(tmp_path / "image", LIMIT_MODULES)

    (_, meta), _ = compile_aiter(tmp_path, base, {"csrc/include/common.h": NEW})

    assert [e.path for e in meta.entries] == ["csrc/include/common.h"]


def test_a_changed_module_config_counts_the_modules_whose_entry_changed(tmp_path):
    base = aiter_image(tmp_path / "image", LIMIT_MODULES + 1)
    config = {f"module_m{i}": {**module_entry(i), "flags_extra_hip": ["'-O1'"]} for i in range(LIMIT_MODULES + 1)}

    with pytest.raises(TooManyModulesError, match=f"touches {LIMIT_MODULES + 1} of aiter's prebuilt modules"):
        compile_aiter(tmp_path, base, {"aiter/jit/optCompilerConfig.json": json.dumps(config)})


def test_a_stack_that_cannot_be_told_from_the_config_is_refused_and_not_passed(tmp_path):
    base = aiter_image(tmp_path / "image", config=False)

    with pytest.raises(PatchApplyError, match="cannot tell which of aiter's modules the patch touches"):
        compile_aiter(tmp_path, base, {"csrc/kernels/m0.cu": NEW})
