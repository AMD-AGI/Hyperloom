"""Bridge skill-directory layout to standard Python package layout.

The skill lives at ``inference-optimizer/`` (hyphen — matches the
``marathon/`` / ``kernel-agent/`` siblings under the Hyperloom repo
root). The Python module name must be ``inference_optimizer`` (underscore
— Python identifier). ``setuptools.find_packages`` cannot compose the
hyphen→underscore mapping through ``package_dir`` alone, so we compute
``packages`` + ``package_dir`` here.

Project metadata (name / version / deps / package-data / pytest config)
still lives in ``pyproject.toml``.
"""

from __future__ import annotations

from setuptools import find_packages, setup

SKILL_DIR = "inference-optimizer"
PKG_NAME = "inference_optimizer"


def _build_package_map() -> tuple[list[str], dict[str, str]]:
    sub_packages = find_packages(where=SKILL_DIR)
    packages = [PKG_NAME] + [f"{PKG_NAME}.{sub}" for sub in sub_packages]

    package_dir: dict[str, str] = {PKG_NAME: SKILL_DIR}
    for sub in sub_packages:
        rel = sub.replace(".", "/")
        package_dir[f"{PKG_NAME}.{sub}"] = f"{SKILL_DIR}/{rel}"
    return packages, package_dir


_packages, _package_dir = _build_package_map()

setup(
    packages=_packages,
    package_dir=_package_dir,
)
