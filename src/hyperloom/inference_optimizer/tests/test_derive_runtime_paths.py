# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for preflight PATH/LD_LIBRARY_PATH derivation (replaces hyperloom.env.sh)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.cli import preflight as cli_preflight


@pytest.fixture(autouse=True)
def _restore_environment():
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.mark.parametrize("framework", [None, "atom", "sglang", "xdit", "custom"])
def test_other_frameworks_do_not_activate_the_vllm_venv(monkeypatch, framework):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("VLLM_VENV_ROOT", "/unrelated/vllm")
    monkeypatch.delenv("FRAMEWORK", raising=False)
    if framework is not None:
        monkeypatch.setenv("FRAMEWORK", framework)

    cli_preflight._derive_runtime_paths()

    assert str(Path("/unrelated/vllm") / "bin") not in os.environ["PATH"].split(os.pathsep)
    assert os.environ["VLLM_VENV_ROOT"] == "/unrelated/vllm"


@pytest.mark.parametrize("framework", [None, "atom", "vllm"])
@pytest.mark.parametrize("environment_framework", [None, "sglang"])
def test_preflight_derives_paths_from_the_actual_framework_argument(
    monkeypatch, tmp_path, framework, environment_framework
):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(tmp_path / "kernel"))
    monkeypatch.delenv("KERNEL_AGENT_ENV", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    monkeypatch.delenv("FRAMEWORK", raising=False)
    if environment_framework is not None:
        monkeypatch.setenv("FRAMEWORK", environment_framework)
    monkeypatch.setenv("VLLM_VENV_ROOT", "/isolated/vllm")
    monkeypatch.setenv("PATH", "/usr/bin")

    class ReachedCredentials(Exception):
        pass

    def stop_before_external_work():
        raise ReachedCredentials

    monkeypatch.setattr(cli_preflight, "_validate_credentials", stop_before_external_work)
    with pytest.raises(ReachedCredentials):
        cli_preflight._preflight(SimpleNamespace(framework=framework))

    expected_framework = framework or environment_framework
    assert os.environ.get("FRAMEWORK") == expected_framework
    assert (str(Path("/isolated/vllm") / "bin") in os.environ["PATH"].split(os.pathsep)) == (
        expected_framework == "vllm"
    )


@pytest.mark.parametrize("layout", ["target", "src"])
def test_runtime_paths_make_known_source_roots_importable_by_children(monkeypatch, tmp_path, layout):
    root = tmp_path / "installed checkout"
    source = root / "src" if layout == "src" else root
    source.mkdir(parents=True)
    (source / "hyperloom").mkdir()
    (source / "runtime_checkout_probe.py").write_text("VALUE = 'checkout'\n", encoding="utf-8")
    magpie = tmp_path / "Magpie checkout"
    magpie.mkdir()
    (magpie / "runtime_magpie_probe.py").write_text("VALUE = 'magpie'\n", encoding="utf-8")
    monkeypatch.setenv("REPO_ROOT", str(root))
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "operator-imports"))

    cli_preflight._derive_runtime_paths()
    cli_preflight._derive_runtime_paths()

    paths = os.environ["PYTHONPATH"].split(os.pathsep)
    assert paths.count(str(source)) == 1
    assert paths.count(str(magpie)) == 1
    assert str(tmp_path / "operator-imports") in paths
    child = subprocess.run(
        [sys.executable, "-c", "import runtime_checkout_probe, runtime_magpie_probe"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("mode", ["baremetal", "docker"])
def test_preflight_runtime_file_reaches_children_without_shell_source(monkeypatch, tmp_path, mode):
    repo = Path(cli_preflight.__file__).resolve().parents[4]
    magpie = tmp_path / "Magpie checkout"
    magpie.mkdir()
    (magpie / "runtime_child_probe.py").write_text("VALUE = 'runtime-child'\n", encoding="utf-8")
    runtime = tmp_path / "kernel-agent.env.sh"
    runtime.write_text(
        f"HYPERLOOM_KERNEL_AGENT_ROOT=/installed/kernel\nMAGPIE_PATH='{magpie}'\n"
        "PYTHON=/host/python\nVIRTUAL_ENV=/host/venv\nINFERENCE_OPTIMIZER_FORCE_PYTHON=1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REPO_ROOT", str(repo))
    monkeypatch.setenv("KERNEL_AGENT_ENV", str(runtime))
    monkeypatch.setenv("HYPERLOOM_RUN_MODE", mode)
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("VIRTUAL_ENV", "")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FORCE_PYTHON", "1")
    for key in ("HYPERLOOM_KERNEL_AGENT_ROOT", "MAGPIE_PATH", "PYTHONPATH"):
        monkeypatch.delenv(key, raising=False)

    class ReachedCredentials(Exception):
        pass

    def check_child_before_external_work():
        child = subprocess.run(
            [sys.executable, "-c", "import hyperloom, runtime_child_probe; print(runtime_child_probe.VALUE)"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.strip() == "runtime-child"
        assert os.environ["PYTHON"] == sys.executable
        assert os.environ["VIRTUAL_ENV"] == ""
        raise ReachedCredentials

    monkeypatch.setattr(cli_preflight, "_validate_credentials", check_child_before_external_work)
    with pytest.raises(ReachedCredentials):
        cli_preflight._preflight(SimpleNamespace(framework="atom"))


@pytest.mark.parametrize("package_root", ["site-packages", "dist-packages"])
def test_runtime_paths_do_not_cross_contaminate_site_packages(monkeypatch, tmp_path, package_root):
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / package_root))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "operator-imports"))

    cli_preflight._derive_runtime_paths()

    assert str(tmp_path / package_root) not in os.environ["PYTHONPATH"].split(os.pathsep)


def test_prepend_path_adds_and_dedups(monkeypatch):
    monkeypatch.setenv("PATH", os.pathsep.join(("/usr/bin", "/bin")))
    cli_preflight._prepend_path("PATH", "/opt/x/bin")
    assert os.environ["PATH"] == f"/opt/x/bin{os.pathsep}/usr/bin{os.pathsep}/bin"
    # Idempotent: already leading -> unchanged.
    cli_preflight._prepend_path("PATH", "/opt/x/bin")
    assert os.environ["PATH"] == f"/opt/x/bin{os.pathsep}/usr/bin{os.pathsep}/bin"
    # Existing-but-not-leading -> moved to front, not duplicated.
    cli_preflight._prepend_path("PATH", "/bin")
    assert os.environ["PATH"] == f"/bin{os.pathsep}/opt/x/bin{os.pathsep}/usr/bin"


def test_prepend_path_empty_entry_noop(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    cli_preflight._prepend_path("PATH", "")
    assert os.environ["PATH"] == "/usr/bin"


def test_derive_runtime_paths_rocm_and_venv(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)

    cli_preflight._derive_runtime_paths()

    path_parts = os.environ["PATH"].split(os.pathsep)
    # ROCm prepended last of the two -> leads; venv follows; system last.
    assert path_parts[0] == str(Path("/opt/rocm") / "bin")
    assert str(Path("/venv") / "bin") in path_parts
    assert os.environ["LD_LIBRARY_PATH"].startswith(str(Path("/opt/rocm") / "lib"))


def test_derive_runtime_paths_isolated_vllm_leads(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm")
    monkeypatch.setenv("VLLM_VENV_ROOT", "/opt/vllm-venv")
    monkeypatch.setenv("FRAMEWORK", "vllm")

    cli_preflight._derive_runtime_paths()

    # vLLM venv prepended last -> must lead PATH.
    assert os.environ["PATH"].split(os.pathsep)[0] == str(Path("/opt/vllm-venv") / "bin")


def test_derive_runtime_paths_noop_without_roots(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    ld_before = os.environ.get("LD_LIBRARY_PATH")

    cli_preflight._derive_runtime_paths()

    assert os.environ["PATH"] == "/usr/bin"
    # No ROCM_PATH -> LD_LIBRARY_PATH must be untouched.
    assert os.environ.get("LD_LIBRARY_PATH") == ld_before


def test_rocm_sdk_wheel_lib_dirs_detects_therock_layout(tmp_path, monkeypatch):
    core_root = tmp_path / "_rocm_sdk_core"
    core_lib = core_root / "lib"
    core_host_math_lib = core_lib / "host-math" / "lib"
    core_sysdeps_lib = core_lib / "rocm_sysdeps" / "lib"
    core_host_math_lib.mkdir(parents=True)
    core_sysdeps_lib.mkdir(parents=True)
    devel_root = tmp_path / "_rocm_sdk_devel"
    devel_lib = devel_root / "lib"
    devel_host_math_lib = devel_lib / "host-math" / "lib"
    devel_host_math_lib.mkdir(parents=True)

    def fake_find_spec(name):
        origins = {
            "_rocm_sdk_core": core_root / "__init__.py",
            "_rocm_sdk_devel": devel_root / "__init__.py",
        }
        origin = origins.get(name)
        return None if origin is None else SimpleNamespace(origin=str(origin))

    monkeypatch.setattr(cli_preflight.importlib.util, "find_spec", fake_find_spec)

    dirs = cli_preflight._rocm_sdk_wheel_lib_dirs()

    assert str(devel_lib) in dirs
    assert str(devel_host_math_lib) in dirs
    assert str(core_lib) in dirs
    assert str(core_host_math_lib) in dirs
    assert str(core_sysdeps_lib) in dirs


def test_rocm_sdk_wheel_lib_dirs_detects_libraries_package(tmp_path, monkeypatch):
    # Real TheRock layout seen on rocm10/gfx950 images: _rocm_sdk_core +
    # _rocm_sdk_libraries only, no _rocm_sdk_devel. The math/DNN libraries
    # (MIOpen, rocBLAS, hipBLASLt, RCCL, ...) live under _rocm_sdk_libraries.
    core_root = tmp_path / "_rocm_sdk_core"
    core_lib = core_root / "lib"
    core_lib.mkdir(parents=True)
    libraries_root = tmp_path / "_rocm_sdk_libraries"
    libraries_lib = libraries_root / "lib"
    libraries_lib.mkdir(parents=True)

    def fake_find_spec(name):
        origins = {
            "_rocm_sdk_core": core_root / "__init__.py",
            "_rocm_sdk_libraries": libraries_root / "__init__.py",
        }
        origin = origins.get(name)
        return None if origin is None else SimpleNamespace(origin=str(origin))

    monkeypatch.setattr(cli_preflight.importlib.util, "find_spec", fake_find_spec)

    dirs = cli_preflight._rocm_sdk_wheel_lib_dirs()

    assert str(core_lib) in dirs
    assert str(libraries_lib) in dirs


def test_rocm_sdk_wheel_lib_dirs_absent_on_standard_rocm_image(monkeypatch):
    # No _rocm_sdk_core/_rocm_sdk_devel packages installed at all.
    monkeypatch.setattr(cli_preflight.importlib.util, "find_spec", lambda name: None)

    assert cli_preflight._rocm_sdk_wheel_lib_dirs() == []


def test_derive_runtime_paths_adds_rocm_sdk_wheel_dirs(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("VLLM_VENV_ROOT", raising=False)
    monkeypatch.setattr(
        cli_preflight,
        "_rocm_sdk_wheel_lib_dirs",
        lambda: ["/opt/venv/_rocm_sdk_core/lib", "/opt/venv/_rocm_sdk_devel/lib/host-math/lib"],
    )

    cli_preflight._derive_runtime_paths()

    ld_parts = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
    assert "/opt/venv/_rocm_sdk_core/lib" in ld_parts
    assert "/opt/venv/_rocm_sdk_devel/lib/host-math/lib" in ld_parts


def test_derive_runtime_paths_noop_when_no_rocm_sdk_wheel(monkeypatch):
    # Standard /opt/rocm image without the TheRock wheel packages -> detection
    # must no-op, preserving pre-fix behavior exactly.
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(cli_preflight, "_rocm_sdk_wheel_lib_dirs", lambda: [])

    cli_preflight._derive_runtime_paths()

    assert os.environ.get("LD_LIBRARY_PATH") is None
