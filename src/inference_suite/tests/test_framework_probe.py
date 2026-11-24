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


def _editable(root: Path) -> dict[str, str]:
    """An install the way ``pip install -e`` lays a flat checkout: a finder appended to ``sys.meta_path`` maps the
    package to its source, and nothing puts the checkout itself on ``sys.path``."""
    source = root / "checkout" / "python" / _PACKAGE
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    site = root / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import importlib.util, sys\n"
        "class _Editable:\n"
        "    @staticmethod\n"
        "    def find_spec(name, path=None, target=None):\n"
        f"        if name != {_PACKAGE!r}:\n"
        "            return None\n"
        f"        return importlib.util.spec_from_file_location(name, {str(source / '__init__.py')!r},\n"
        f"            submodule_search_locations=[{str(source)!r}])\n"
        "sys.meta_path.append(_Editable)\n"
    )
    return {"PYTHONPATH": str(site), "PATH": os.environ.get("PATH", "")}


def test_the_install_names_the_interpreter_package_origin_and_version(tmp_path: Path) -> None:
    install = probe_install(sys.executable, Framework.XDIT, _site(tmp_path, version="0.5.23.dev4"), cwd=tmp_path)
    assert install.executable == os.path.realpath(sys.executable)
    assert install.python == ".".join(str(part) for part in sys.version_info[:3])
    assert install.origin == str((tmp_path / _PACKAGE / "__init__.py").resolve())
    assert install.version == "0.5.23.dev4"


def test_the_install_names_where_each_requested_package_is_imported_from(tmp_path: Path) -> None:
    env = _site(tmp_path, version="0.1")
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "__init__.py").write_text("")
    (tmp_path / "single.py").write_text("")

    install = probe_install(
        sys.executable, Framework.XDIT, env, cwd=tmp_path, packages=("kernels", "single", "not_installed")
    )

    assert install.import_roots == {
        _PACKAGE: str((tmp_path / _PACKAGE).resolve()),
        "kernels": str((tmp_path / "kernels").resolve()),
        "single": str((tmp_path / "single.py").resolve()),
    }


def test_a_package_with_no_distribution_metadata_has_no_version(tmp_path: Path) -> None:
    assert probe_install(sys.executable, Framework.XDIT, _site(tmp_path, version=None), cwd=tmp_path).version is None


def test_a_checkout_in_the_callers_directory_does_not_shadow_an_editable_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sglang image's working directory holds the checkout /sgl-workspace/sglang, which has no __init__.py: started
    # there, ``python -c`` finds it first as a namespace package and never asks the editable finder.
    env = _editable(tmp_path)
    workspace = tmp_path / "workspace"
    (workspace / _PACKAGE).mkdir(parents=True)
    monkeypatch.chdir(workspace)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    install = probe_install(sys.executable, Framework.XDIT, env, cwd=run_dir)

    assert install.origin == str((tmp_path / "checkout" / "python" / _PACKAGE / "__init__.py").resolve())
    assert install.import_roots == {_PACKAGE: str((tmp_path / "checkout" / "python" / _PACKAGE).resolve())}
    with pytest.raises(FrameworkProbeError, match=f"cannot import {_PACKAGE}"):
        probe_install(sys.executable, Framework.XDIT, env, cwd=workspace)


def test_a_framework_the_interpreter_cannot_import_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FrameworkProbeError, match="cannot import atom"):
        probe_install(sys.executable, Framework.ATOM, _site(tmp_path, version="0.1"), cwd=tmp_path)


def test_a_framework_that_ships_no_package_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FrameworkProbeError, match="ships no package"):
        probe_install(sys.executable, Framework.CUSTOM, {}, cwd=tmp_path)


def test_a_missing_interpreter_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FrameworkProbeError, match="could not run the probe"):
        probe_install(str(tmp_path / "python"), Framework.XDIT, {}, cwd=tmp_path)


def test_a_probe_that_prints_no_object_names_its_stderr(tmp_path: Path) -> None:
    program = "import sys; print('[1]'); sys.stderr.write('no object here')"
    with pytest.raises(FrameworkProbeError, match="no object here"):
        run_probe(sys.executable, program, (), {}, cwd=tmp_path)


def test_a_failing_probe_is_an_error_even_after_printing_an_object(tmp_path: Path) -> None:
    program = "import sys; print('{}'); sys.exit(3)"
    with pytest.raises(FrameworkProbeError, match="exited 3"):
        run_probe(sys.executable, program, (), {}, cwd=tmp_path)


def test_the_probe_answer_is_its_last_printed_object(tmp_path: Path) -> None:
    program = "import json, sys; print('banner'); print(json.dumps({'args': sys.argv[1:]}))"
    assert run_probe(sys.executable, program, ("a", "b"), {}, cwd=tmp_path) == {"args": ["a", "b"]}


def _native_server(root: Path, printed: str, *, code: int = 0) -> dict[str, str]:
    """A stand-in ``llama-server`` on a PATH of its own, printing ``printed`` to stderr as llama.cpp does."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    program = bin_dir / "llama-server"
    program.write_text(f"#!/bin/sh\nprintf %s {printed!r} >&2\nexit {code}\n")
    program.chmod(0o755)
    return {"PATH": str(bin_dir)}


def test_a_native_server_reports_its_own_version_whatever_the_interpreter(tmp_path: Path) -> None:
    env = _native_server(tmp_path, "version: 1.0.0-dev (build 12439, commit 8bfd64b5a)\nbuilt with GNU 13.3.0\n")
    install = probe_install(str(tmp_path / "no-python"), Framework.LLAMACPP, env, cwd=tmp_path)

    assert install.version == "1.0.0-dev"
    assert install.executable == install.origin == str((tmp_path / "bin" / "llama-server").resolve())


def test_a_native_server_missing_from_the_launch_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FrameworkProbeError, match="not on the launch PATH"):
        probe_install(sys.executable, Framework.LLAMACPP, {"PATH": str(tmp_path)}, cwd=tmp_path)


def test_a_native_server_that_prints_no_version_is_an_error(tmp_path: Path) -> None:
    env = _native_server(tmp_path, "error: unknown argument", code=1)
    with pytest.raises(FrameworkProbeError, match="exited 1 with no version: error: unknown argument"):
        probe_install(sys.executable, Framework.LLAMACPP, env, cwd=tmp_path)
