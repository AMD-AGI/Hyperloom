# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The framework checkout a session runs on, read as its root :class:`~hyperloom.contracts.recipe.Recipe`.

The root is the checkout the operator named in :data:`MB_FRAMEWORK_REPO_ENV`, at its ``HEAD``, launched as the main
flow launches it, without a patch (:func:`launched_root`); a framework installed as a package, not a git checkout, is
pinned to nothing.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches, Recipe, RecipeError
from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    detect_framework_version,
)

#: How long one read-only git command may take before the checkout counts as unreadable.
GIT_TIMEOUT_SECONDS = 60.0

#: The framework checkout the operator names, the variable the main flow seeds ``framework_repo_path`` from
#: (src/hyperloom/inference_optimizer/cli/bootstrap.py).
MB_FRAMEWORK_REPO_ENV = "FRAMEWORK_REPO_PATH"
#: The framework version the main flow records when its command line names none, before it detects one
#: (``resolve_framework_version`` of src/hyperloom/inference_optimizer/cli/bootstrap.py, which the scheduler may not
#: import: it loads the main flow's command line).
MB_FRAMEWORK_VERSION_ENV = "FRAMEWORK_VERSION"
DEFAULT_REMOTE = "origin"


class PreludeReadError(RuntimeError):
    """The framework checkout cannot be read, or holds something the root recipe cannot express."""


@dataclass(frozen=True)
class UpstreamPin:
    """Where a component's source comes from: its git remote and the commit the patches apply on."""

    repo_url: str
    ref: str


def launched_root(framework: str, env: Mapping[str, str], base_image_digest: str) -> Recipe:
    """The root recipe of a session run with ``env``: the framework checkout ``env`` names, at its ``HEAD``, on
    ``base_image_digest``. Raises :class:`PreludeReadError`. Blocks."""

    return root_recipe(framework, framework_pin(_checkout(env.get(MB_FRAMEWORK_REPO_ENV, ""))), base_image_digest)


def launched_framework_version(framework: str, env: Mapping[str, str]) -> str:
    """The version of ``framework`` a session run with ``env`` records, as the main flow resolves it for a command
    line that names none: :data:`MB_FRAMEWORK_VERSION_ENV`, else the installed package's, ``""`` when that cannot be
    told. Blocks: detecting it imports the framework's package."""

    named = env.get(MB_FRAMEWORK_VERSION_ENV, "").strip()
    if named:
        return named
    detected = detect_framework_version(framework)
    return "" if detected == DEFAULT_FRAMEWORK_VERSION_SLUG else detected


def framework_pin(root: Path | None) -> UpstreamPin | None:
    """The framework checkout ``root``'s remote at its ``HEAD``; ``None`` without a checkout."""

    if root is None:
        return None
    remotes = _git(root, "remote").split()
    if not remotes:
        raise PreludeReadError(f"{root} has no git remote to pin the framework to")
    remote = DEFAULT_REMOTE if DEFAULT_REMOTE in remotes else remotes[0]
    return UpstreamPin(repo_url=_git(root, "remote", "get-url", remote), ref=_git(root, "rev-parse", "HEAD"))


def root_recipe(framework: str, pin: UpstreamPin | None, base_image_digest: str) -> Recipe:
    """The baseline as a recipe on ``base_image_digest``: the main flow's own launch, and ``framework`` at ``pin``
    without a patch."""

    try:
        components = () if pin is None else (ComponentPatches(framework, pin.repo_url, pin.ref),)
        return Recipe(base_image_digest=base_image_digest, components=components)
    except RecipeError as exc:
        raise PreludeReadError(f"the framework checkout is not a valid root recipe: {exc}") from exc


def _checkout(path: str) -> Path | None:
    root = Path(path.strip()) if path.strip() else None
    return root if root is not None and _is_checkout(root) else None


def _is_checkout(root: Path) -> bool:
    try:
        _git(root, "rev-parse", "--show-toplevel")
    except PreludeReadError:
        return False
    return True


def _git(root: Path, *args: str) -> str:
    """The stripped standard output of ``git -C root args``. Blocks."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args), capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_SECONDS
        )
    except subprocess.CalledProcessError as exc:
        raise PreludeReadError(f"git {' '.join(args)} in {root} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreludeReadError(f"git {' '.join(args)} in {root} failed: {exc}") from exc
    return completed.stdout.strip()


__all__ = [
    "MB_FRAMEWORK_REPO_ENV",
    "MB_FRAMEWORK_VERSION_ENV",
    "PreludeReadError",
    "UpstreamPin",
    "framework_pin",
    "launched_framework_version",
    "launched_root",
    "root_recipe",
]
