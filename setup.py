# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Setuptools bridge for project metadata in ``pyproject.toml``."""

from __future__ import annotations

from setuptools import find_packages, setup

# tree-reform.MD P2.4: ``inference_optimizer`` was promoted from a repo-root
# flat-layout package to ``src/hyperloom/inference_optimizer`` (alongside the
# already-promoted ``orchestrator`` and ``common``), so the whole ``hyperloom``
# distribution is now a single src-layout namespace discovered from ``src/``.
setup(
    packages=find_packages(where="src"),
    package_dir={"hyperloom": "src/hyperloom"},
)
