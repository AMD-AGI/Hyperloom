# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Patch grounding must survive a checkout owned by another uid."""

from __future__ import annotations

import subprocess

import pytest

from hyperloom.orchestrator.specialists import patch_safety

_PATCH = """--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+hello world
"""


@pytest.fixture
def foreign_checkout(tmp_path, monkeypatch):
    """A real checkout the patch applies to, turned foreign after setup."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t.local",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    return repo


def _verdict(tmp_path, checkout) -> str:
    patch = tmp_path / "hello.patch"
    patch.write_text(_PATCH, encoding="utf-8")
    _kept, _ungrounded, grounding, _spans = patch_safety.vet_patches([str(patch)], base_checkout=checkout)
    return grounding[str(patch)]


def test_applicable_patch_is_not_called_stale(tmp_path, foreign_checkout):
    """The worst shape of this bug: a wrong verdict rather than an error."""
    assert _verdict(tmp_path, foreign_checkout) == patch_safety.GROUND_APPLIES


def test_a_genuinely_stale_patch_is_still_stale(tmp_path, foreign_checkout):
    """Lifting the refusal must not turn grounding into a rubber stamp."""
    (foreign_checkout / "hello.txt").write_text("something else\n", encoding="utf-8")

    assert _verdict(tmp_path, foreign_checkout) == patch_safety.GROUND_STALE
