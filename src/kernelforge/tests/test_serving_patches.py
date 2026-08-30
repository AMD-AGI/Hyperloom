"""Tests for the versioned serving patch assets.

These guard the interface contract that the Hyperloom applier depends on:
the directory layout, the manifest, and the presence of a patch file for every
supported version.
"""

from __future__ import annotations

from kernelforge.resources import resource_path

SERVING_PATCHES = resource_path("serving_patches")
SGLANG_DIR = SERVING_PATCHES / "sglang"
MANIFEST = SGLANG_DIR / "SUPPORTED_VERSIONS.txt"


def _versioned_subdir_name(version: str) -> str:
    """Mirror Hyperloom's `_versioned_patches_subdir_name` convention."""
    return "sglang_" + version.replace(".", "_")


def _supported_versions() -> list[str]:
    versions = []
    for raw in MANIFEST.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            versions.append(line)
    return versions


def test_manifest_exists_and_lists_0_5_12():
    assert MANIFEST.is_file()
    assert "0.5.12" in _supported_versions()


def test_every_supported_version_has_patch():
    versions = _supported_versions()
    assert versions, "manifest must list at least one version"
    for version in versions:
        patch = SGLANG_DIR / _versioned_subdir_name(version) / "fp8_blockscale_ck_routing.patch"
        assert patch.is_file(), f"missing patch for sglang {version}: {patch}"


def test_subdir_name_convention():
    assert _versioned_subdir_name("0.5.12") == "sglang_0_5_12"


def test_patch_is_git_format_and_targets_fp8_utils():
    patch = SGLANG_DIR / "sglang_0_5_12" / "fp8_blockscale_ck_routing.patch"
    text = patch.read_text()
    assert text.startswith("diff --git ")
    assert "python/sglang/srt/layers/quantization/fp8_utils.py" in text
    assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" in text


def test_readme_exists():
    assert (SERVING_PATCHES / "README.md").is_file()
