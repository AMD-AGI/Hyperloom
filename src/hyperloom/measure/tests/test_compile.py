# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.contracts.recipe import (
    ComponentChange,
    ConfigDelta,
    KernelArtifact,
    KernelKind,
    PatchRef,
    Recipe,
    RecipeDelta,
    RecipeError,
)
from hyperloom.measure.compile import (
    BaseImageMismatchError,
    BlobError,
    CompileError,
    ComponentLayout,
    EnvironmentCompiler,
    KernelLoadError,
    LocalBlobStore,
    PackageBase,
    PatchApplyError,
    UnsupportedComponentError,
    compile_environment,
    locate_installed_package,
)
from hyperloom.measure.tests._recipes import IMAGE, PREFIX, git_patch, put_patch, recipe_with, sha, write_tree

OLD_A = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
NEW_A = "def f():\n    return 10\n\n\ndef g():\n    return 2\n"


@pytest.fixture()
def blobs(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture()
def base(tmp_path: Path) -> PackageBase:
    root = write_tree(tmp_path / "image" / "sglang", {"srt/a.py": OLD_A, "srt/old.py": "x = 1\n", "__init__.py": ""})
    return PackageBase("sglang", root, "sglang", "0.5.4")


def locator_for(base: PackageBase, calls: list[ComponentLayout] | None = None):
    def locate(layout: ComponentLayout) -> PackageBase:
        if calls is not None:
            calls.append(layout)
        return base

    return locate


def compile_with(recipe, delta, blobs, base, **kwargs):
    return compile_environment(
        recipe, delta, blobs=blobs, active_image_digest=IMAGE, locator=locator_for(base), **kwargs
    )


def patch_of(tmp_path: Path, name: str, before, after) -> bytes:
    prefixed = lambda files: {f"{PREFIX}/{rel}": text for rel, text in files.items()}  # noqa: E731
    return git_patch(tmp_path / name, prefixed(before), prefixed(after))


def test_patch_becomes_an_upsert_entry_and_the_base_stays_untouched(tmp_path, blobs, base):
    patch = patch_of(tmp_path, "p1", {"srt/a.py": OLD_A}, {"srt/a.py": NEW_A})
    recipe = recipe_with((put_patch(blobs, "a.patch", patch),))

    env = compile_with(recipe, RecipeDelta(), blobs, base)

    (root,) = env.roots
    assert (root.id, root.import_name, root.dist_version) == ("sglang", "sglang", "0.5.4")
    (entry,) = root.entries
    assert (entry.path, entry.op, entry.mode) == ("srt/a.py", "upsert", "0644")
    assert blobs.get(entry.blob) == NEW_A.encode()
    assert (base.path / "srt/a.py").read_text() == OLD_A
    assert env.to_document()["roots"][0]["entries"][0]["blob"] == f"sha256:{sha(NEW_A.encode())}"


def test_stack_applies_in_order_and_records_creates_and_deletes(tmp_path, blobs, base):
    first = patch_of(tmp_path, "p1", {"srt/a.py": OLD_A}, {"srt/a.py": NEW_A, "srt/new.py": "y = 2\n"})
    second = patch_of(
        tmp_path, "p2", {"srt/a.py": NEW_A, "srt/old.py": "x = 1\n"}, {"srt/a.py": "z = 3\n", "srt/old.py": None}
    )
    refs = (put_patch(blobs, "1.patch", first), put_patch(blobs, "2.patch", second))

    env = compile_with(recipe_with(refs), RecipeDelta(), blobs, base)

    entries = {entry.path: entry for entry in env.roots[0].entries}
    assert set(entries) == {"srt/a.py", "srt/new.py", "srt/old.py"}
    assert blobs.get(entries["srt/a.py"].blob) == b"z = 3\n"
    assert entries["srt/old.py"].op == "delete"


def test_second_patch_alone_does_not_apply_so_order_and_content_matter(tmp_path, blobs, base):
    second = patch_of(tmp_path, "p2", {"srt/a.py": NEW_A}, {"srt/a.py": "z = 3\n"})
    recipe = recipe_with((put_patch(blobs, "2.patch", second),))

    with pytest.raises(PatchApplyError, match="does not apply to the installed package"):
        compile_with(recipe, RecipeDelta(), blobs, base)


def test_a_patch_made_against_other_source_is_refused_not_skipped(tmp_path, blobs, base):
    drifted = patch_of(tmp_path, "p", {"srt/a.py": "something else\n"}, {"srt/a.py": "patched\n"})
    recipe = recipe_with((put_patch(blobs, "d.patch", drifted),))

    with pytest.raises(PatchApplyError, match="d.patch does not apply"):
        compile_with(recipe, RecipeDelta(), blobs, base)


def test_a_patch_that_reaches_outside_the_package_is_refused(tmp_path, blobs, base):
    outside = git_patch(tmp_path / "p", {"test/t.py": "a\n"}, {"test/t.py": "b\n"})
    recipe = recipe_with((put_patch(blobs, "o.patch", outside),))

    with pytest.raises(PatchApplyError, match="outside the installed package"):
        compile_with(recipe, RecipeDelta(), blobs, base)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", "not a git patch"),
        (b"diff --git a/python/sglang/x.bin b/python/sglang/x.bin\nGIT binary patch\nliteral 0\n", "binary"),
        (b"diff --git a/python/sglang/l b/python/sglang/l\nnew file mode 120000\n", "symbolic links"),
        (b"diff --git a/python/sglang/../../etc/x b/python/sglang/../../etc/x\n", "unsafe path"),
        (b"\xff\xfe", "not UTF-8"),
    ],
)
def test_patches_that_cannot_be_applied_faithfully_are_refused(blobs, base, data, message):
    recipe = recipe_with((put_patch(blobs, "x.patch", data),))

    with pytest.raises(PatchApplyError, match=message):
        compile_with(recipe, RecipeDelta(), blobs, base)


def test_a_patch_blob_that_is_missing_or_tampered_is_refused(tmp_path, blobs, base):
    patch = patch_of(tmp_path, "p", {"srt/a.py": OLD_A}, {"srt/a.py": NEW_A})
    ref = put_patch(blobs, "a.patch", patch)
    with pytest.raises(BlobError, match="not in the blob store"):
        compile_with(recipe_with((PatchRef(ref.path, "c" * 64),)), RecipeDelta(), blobs, base)

    blob = blobs.root / ref.sha256
    blob.chmod(0o644)
    blob.write_bytes(patch + b"\n")
    with pytest.raises(BlobError, match="does not hash to its address"):
        compile_with(recipe_with((ref,)), RecipeDelta(), blobs, base)


def test_blobs_are_written_once_and_read_only(blobs):
    address = blobs.put(b"data")
    assert blobs.put(b"data") == address
    assert (blobs.root / address).stat().st_mode & 0o222 == 0


@pytest.mark.parametrize(
    "recipe",
    [
        Recipe(base_image_digest="sha256:" + "d" * 64),
        Recipe(base_image_digest=IMAGE, runtime_snapshot="sha256:" + "d" * 64),
    ],
)
def test_other_base_image_is_refused(blobs, base, recipe):
    with pytest.raises(BaseImageMismatchError, match="is not the active framework image"):
        compile_with(recipe, RecipeDelta(), blobs, base)


def test_digest_with_repository_name_matches_the_active_image(blobs, base):
    recipe = Recipe(base_image_digest=f"registry.example/fw@{IMAGE}", runtime_snapshot=IMAGE)

    assert compile_with(recipe, RecipeDelta(), blobs, base).is_base


def test_delta_that_does_not_fit_is_a_recipe_error_from_the_contract(blobs, base):
    delta = RecipeDelta(components=(ComponentChange("sglang", remove_patches=("patch/none.patch",)),))

    with pytest.raises(RecipeError, match="has no patches"):
        compile_with(recipe_with(), delta, blobs, base)


def test_every_refusal_is_a_recipe_error():
    for error in (PatchApplyError, BaseImageMismatchError, UnsupportedComponentError, BlobError, KernelLoadError):
        assert issubclass(error, CompileError) and issubclass(error, RecipeError)


def test_component_without_a_layout_is_refused(tmp_path, blobs, base):
    patch = patch_of(tmp_path, "p", {"srt/a.py": OLD_A}, {"srt/a.py": NEW_A})
    recipe = recipe_with((put_patch(blobs, "a.patch", patch),), component="mystery")

    with pytest.raises(UnsupportedComponentError, match="no package layout"):
        compile_with(recipe, RecipeDelta(), blobs, base)


def kernel(**metadata) -> KernelArtifact:
    return KernelArtifact(KernelKind.GEMM, "bf16", "kernel/gemm.csv", sha("table".encode()), metadata)


LOAD = {"load": {"via": "env_file", "env": "AITER_CONFIG_GEMM_BF16"}}


def test_gemm_table_is_attached_and_loaded_through_its_variable(blobs, base):
    blobs.put(b"table")
    delta = RecipeDelta(add_kernels=(kernel(**LOAD),))

    env = compile_with(recipe_with(), delta, blobs, base)

    (attachment,) = env.attachments
    assert attachment == AttachmentRef(attachment.id, "file", sha(b"table"))
    assert env.envs == {"AITER_CONFIG_GEMM_BF16": "${HL_ATTACH:" + attachment.id + "}"}
    assert env.recipe.kernels == (kernel(**LOAD),)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (kernel(), "metadata.load is missing"),
        (kernel(load={"via": "dir"}), "unknown load via 'dir'"),
        (kernel(load={"via": "env_file", "env": "not a name"}), "must name an environment variable"),
        (
            KernelArtifact(KernelKind.FUSION, "f", "kernel/f.py", sha(b"table"), LOAD),
            "no loading convention for fusion",
        ),
    ],
)
def test_kernel_without_a_supported_way_to_load_is_refused(blobs, base, artifact, message):
    blobs.put(b"table")

    with pytest.raises(KernelLoadError, match=message):
        compile_with(recipe_with(), RecipeDelta(add_kernels=(artifact,)), blobs, base)


def test_kernel_blob_must_exist_and_its_variable_must_be_free(blobs, base):
    with pytest.raises(BlobError, match="not in the blob store"):
        compile_with(recipe_with(), RecipeDelta(add_kernels=(kernel(**LOAD),)), blobs, base)

    blobs.put(b"table")
    clash = RecipeDelta(
        config=ConfigDelta(envs={"AITER_CONFIG_GEMM_BF16": "/tmp/mine.csv"}), add_kernels=(kernel(**LOAD),)
    )
    with pytest.raises(KernelLoadError, match="already set"):
        compile_with(recipe_with(), clash, blobs, base)


def test_same_recipe_and_change_compile_once(tmp_path, blobs, base):
    patch = patch_of(tmp_path, "p", {"srt/a.py": OLD_A}, {"srt/a.py": NEW_A})
    recipe = recipe_with((put_patch(blobs, "a.patch", patch),))
    calls: list[ComponentLayout] = []
    compiler = EnvironmentCompiler(blobs, IMAGE, locator_for(base, calls))

    first = compiler.compile(recipe, RecipeDelta())
    assert compiler.compile(recipe, RecipeDelta()) is first
    assert len(calls) == 1
    assert compiler.compile(recipe, RecipeDelta(config=ConfigDelta(server_args="--x 1"))) is not first


def test_locator_accepts_site_packages_and_refuses_trees_outside_the_image(tmp_path, monkeypatch):
    site = write_tree(tmp_path / "lib" / "site-packages", {"w5_installed/__init__.py": ""})
    shared = write_tree(tmp_path / "shared", {"w5_shared/__init__.py": ""})
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(shared))
    try:
        found = locate_installed_package(ComponentLayout("w5_installed", "w5_installed"))
        assert found.path == (site / "w5_installed").resolve()
        with pytest.raises(UnsupportedComponentError, match="neither installed in site-packages"):
            locate_installed_package(ComponentLayout("w5_shared", "w5_shared"))
        with pytest.raises(UnsupportedComponentError, match="is not installed"):
            locate_installed_package(ComponentLayout("w5_absent", "w5_absent"))
    finally:
        for name in ("w5_installed", "w5_shared"):
            sys.modules.pop(name, None)
