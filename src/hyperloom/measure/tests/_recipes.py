# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builders shared by the measurement tests: a fake installed package, real git patches and recipes."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches, PatchRef, Recipe

IMAGE = "sha256:" + "a" * 64
COMMIT = "b" * 40
PREFIX = "python/sglang"


def write_tree(root: Path, files: Mapping[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def git_patch(workdir: Path, before: Mapping[str, str], after: Mapping[str, str | None]) -> bytes:
    """The ``git diff`` that turns ``before`` into ``after`` (``None`` deletes), paths relative to the repo root."""

    workdir.mkdir(parents=True)
    write_tree(workdir, before)
    run = lambda *args: subprocess.run(["git", *args], cwd=workdir, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "base")
    for rel, text in after.items():
        if text is None:
            (workdir / rel).unlink()
        else:
            write_tree(workdir, {rel: text})
    run("add", "-A")
    return subprocess.run(
        ["git", "diff", "--cached", "--no-color"], cwd=workdir, check=True, capture_output=True
    ).stdout


def put_patch(blobs, name: str, data: bytes) -> PatchRef:
    return PatchRef(f"patch/{name}", blobs.put(data))


def recipe_with(patches: tuple[PatchRef, ...] = (), component: str = "sglang", **kwargs) -> Recipe:
    components = (ComponentPatches(component, "https://example.invalid/sglang.git", COMMIT, patches),)
    return Recipe(base_image_digest=IMAGE, components=components, **kwargs)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
