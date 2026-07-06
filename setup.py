# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Setuptools bridge for project metadata in ``pyproject.toml``."""

from __future__ import annotations

from setuptools import find_packages, setup

# Tree-reform transition (see tree-reform.MD, P2.0): discover BOTH the legacy
# flat-layout ``inference_optimizer`` package at the repo root AND the new
# ``src/hyperloom`` src-layout namespace, so both import roots resolve during
# the migration window. ``package_dir`` maps only the new namespace under
# ``src/`` (longest-prefix match), leaving the legacy root layout untouched.
setup(
    packages=(
        find_packages(
            include=[
                "inference_optimizer",
                "inference_optimizer.*",
            ],
        )
        + find_packages(where="src")
    ),
    package_dir={"hyperloom": "src/hyperloom"},
)
