# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Setuptools bridge for project metadata in ``pyproject.toml``."""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    packages=find_packages(include=["inference_optimizer", "inference_optimizer.*"]),
)
