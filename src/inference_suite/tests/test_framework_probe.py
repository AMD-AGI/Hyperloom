# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Probing the serving interpreter: a real interpreter over a stand-in framework package on its path.

The stand-in is xDiT's ``xfuser``, which the test environment does not install, so no installed distribution
answers in its place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from inference_suite.framework_probe import FrameworkProbeError, probe_install, run_probe
from inference_suite.frameworks import Framework

_PACKAGE = "xfuser"


def _site(root: Path, *, version: str | None) -> dict[str, str]:
    package = root / _PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('the probe must not import the framework')\n")
    if version is not None:
        metadata = root / f"{_PACKAGE}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {_PACKAGE}\nVersion: {version}\n")
    return {"PYTHONPATH": str(root), "PATH": os.environ.get("PATH", "")}


def test_the_install_names_the_interpreter_package_origin_and_version(tmp_path: Path) -> None:
    install = probe_install(sys.executable, Framework.XDIT, _site(tmp_path, version="0.5.23.dev4"))
    assert install.executable == os.path.realpath(sys.executable)
    assert install.python == ".".join(str(part) for part in sys.version_info[:3])
    assert install.origin == str((tmp_path / _PACKAGE / "__init__.py").resolve())
    assert install.version == "0.5.23.dev4"


def test_a_package_with_no_distribution_metadata_has_no_version(tmp_path: Path) -> None:
    assert probe_install(sys.executable, Framework.XDIT, _site(tmp_path, version=None)).version is None


def test_a_framework_the_interpreter_cannot_import_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FrameworkProbeError, match="cannot import atom"):
        probe_install(sys.executable, Framework.ATOM, _site(tmp_path, version="0.1"))


def test_a_framework_that_ships_no_package_is_an_error() -> None:
    with pytest.raises(FrameworkProbeError, match="ships no package"):
        probe_install(sys.executable, Framework.CUSTOM, {})


def test_a_missing_interpreter_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FrameworkProbeError, match="could not run the probe"):
        probe_install(str(tmp_path / "python"), Framework.XDIT, {})


def test_a_probe_that_prints_no_object_names_its_stderr() -> None:
    program = "import sys; print('[1]'); sys.stderr.write('no object here')"
    with pytest.raises(FrameworkProbeError, match="no object here"):
        run_probe(sys.executable, program, (), {})


def test_a_failing_probe_is_an_error_even_after_printing_an_object() -> None:
    program = "import sys; print('{}'); sys.exit(3)"
    with pytest.raises(FrameworkProbeError, match="exited 3"):
        run_probe(sys.executable, program, (), {})


def test_the_probe_answer_is_its_last_printed_object() -> None:
    program = "import json, sys; print('banner'); print(json.dumps({'args': sys.argv[1:]}))"
    assert run_probe(sys.executable, program, ("a", "b"), {}) == {"args": ["a", "b"]}
