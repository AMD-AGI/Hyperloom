# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install_baremetal.sh's TheRock ROCm SDK wheel lib-dir detection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_INSTALL_SH = _ASSETS / "install_baremetal.sh"


def _extract_function(name: str) -> str:
    return subprocess.run(
        ["sed", "-n", f"/^{name}()/,/^}}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _run_rocm_sdk_wheel_lib_dirs(pythonpath: str) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("rocm_sdk_wheel_lib_dirs")
    assert fn_src.strip(), "rocm_sdk_wheel_lib_dirs() not found in install_baremetal.sh"
    script = f"set -euo pipefail\n{fn_src}\nrocm_sdk_wheel_lib_dirs '{sys.executable}'\n"
    env = {"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True, env=env)


def test_rocm_sdk_wheel_lib_dirs_detects_therock_layout(tmp_path: Path) -> None:
    site = tmp_path / "site"
    core_lib = site / "_rocm_sdk_core" / "lib"
    core_sysdeps_lib = core_lib / "rocm_sysdeps" / "lib"
    core_sysdeps_lib.mkdir(parents=True)
    (site / "_rocm_sdk_core" / "__init__.py").write_text("")
    devel_host_math_lib = site / "_rocm_sdk_devel" / "lib" / "host-math" / "lib"
    devel_host_math_lib.mkdir(parents=True)
    (site / "_rocm_sdk_devel" / "__init__.py").write_text("")

    result = _run_rocm_sdk_wheel_lib_dirs(str(site))

    assert result.returncode == 0, result.stderr
    dirs = result.stdout.splitlines()
    assert str(core_lib) in dirs
    assert str(core_sysdeps_lib) in dirs
    assert str(devel_host_math_lib) in dirs
    # devel/lib itself is a byproduct of the nested mkdir and must also be reported.
    assert str(site / "_rocm_sdk_devel" / "lib") in dirs


def test_rocm_sdk_wheel_lib_dirs_detects_libraries_package(tmp_path: Path) -> None:
    # Real rocm10/gfx950 layout: _rocm_sdk_core + _rocm_sdk_libraries, no
    # _rocm_sdk_devel. MIOpen/rocBLAS/hipBLASLt/RCCL live under the latter.
    site = tmp_path / "site"
    core_lib = site / "_rocm_sdk_core" / "lib"
    core_lib.mkdir(parents=True)
    (site / "_rocm_sdk_core" / "__init__.py").write_text("")
    libraries_lib = site / "_rocm_sdk_libraries" / "lib"
    libraries_lib.mkdir(parents=True)
    (site / "_rocm_sdk_libraries" / "__init__.py").write_text("")

    result = _run_rocm_sdk_wheel_lib_dirs(str(site))

    assert result.returncode == 0, result.stderr
    dirs = result.stdout.splitlines()
    assert str(core_lib) in dirs
    assert str(libraries_lib) in dirs


def test_rocm_sdk_wheel_lib_dirs_absent_on_standard_rocm_image(tmp_path: Path) -> None:
    # Empty PYTHONPATH dir: neither _rocm_sdk_core nor _rocm_sdk_devel importable,
    # matching a standard /opt/rocm image. Detection must report nothing.
    empty_site = tmp_path / "empty"
    empty_site.mkdir()

    result = _run_rocm_sdk_wheel_lib_dirs(str(empty_site))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def _run_rocm_sdk_wheel_include_dir(pythonpath: str) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("rocm_sdk_wheel_include_dir")
    assert fn_src.strip(), "rocm_sdk_wheel_include_dir() not found in install_baremetal.sh"
    script = f"set -euo pipefail\n{fn_src}\nrocm_sdk_wheel_include_dir '{sys.executable}'\n"
    env = {"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True, env=env)


def test_rocm_sdk_wheel_include_dir_finds_core_package(tmp_path: Path) -> None:
    site = tmp_path / "site"
    hip_include = site / "_rocm_sdk_core" / "include" / "hip"
    hip_include.mkdir(parents=True)
    (site / "_rocm_sdk_core" / "__init__.py").write_text("")

    result = _run_rocm_sdk_wheel_include_dir(str(site))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(site / "_rocm_sdk_core")


def test_rocm_sdk_wheel_include_dir_absent_on_standard_rocm_image(tmp_path: Path) -> None:
    empty_site = tmp_path / "empty"
    empty_site.mkdir()

    result = _run_rocm_sdk_wheel_include_dir(str(empty_site))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def _run_ensure_openmpi_runtime(
    tmp_path: Path, *, ldconfig_has_mpi: bool, uid: int, apt_outcomes: list[bool]
) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("ensure_openmpi_runtime")
    assert fn_src.strip(), "ensure_openmpi_runtime() not found in install_baremetal.sh"
    idx_file = tmp_path / "apt_call_idx"
    outcomes = " ".join("0" if ok else "1" for ok in apt_outcomes)
    stub = f"""
log() {{ echo "LOG: $*"; }}
warn() {{ echo "WARN: $*" >&2; }}
ldconfig() {{ [ "$1" = "-p" ] && {"echo 'libmpi.so.40 => /usr/lib/x86_64-linux-gnu/libmpi.so.40'" if ldconfig_has_mpi else "true"}; }}
apt-get() {{
  if [ "$1" = "update" ]; then return 0; fi
  read -r -a _outcomes <<< "{outcomes}"
  _idx=$(cat "{idx_file}" 2>/dev/null || echo 0)
  echo $((_idx + 1)) > "{idx_file}"
  [ "${{_outcomes[$_idx]:-1}}" = "0" ]
}}
id() {{ echo {uid}; }}
"""
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nensure_openmpi_runtime\n"
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)


def test_ensure_openmpi_runtime_noop_when_already_resolvable(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=True, uid=0, apt_outcomes=[])
    assert result.returncode == 0, result.stderr
    assert "installed" not in result.stdout


def test_ensure_openmpi_runtime_warns_when_not_root(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=1000, apt_outcomes=[])
    assert result.returncode == 0, result.stderr
    assert "not running as root" in result.stderr


def test_ensure_openmpi_runtime_tries_t64_name_first(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=0, apt_outcomes=[True])
    assert result.returncode == 0, result.stderr
    assert "libopenmpi3t64" in result.stdout


def test_ensure_openmpi_runtime_falls_back_to_older_debian_name(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=0, apt_outcomes=[False, True])
    assert result.returncode == 0, result.stderr
    assert "installed libopenmpi3 " in result.stdout


def test_ensure_openmpi_runtime_warns_when_both_names_fail(tmp_path: Path) -> None:
    result = _run_ensure_openmpi_runtime(tmp_path, ldconfig_has_mpi=False, uid=0, apt_outcomes=[False, False])
    assert result.returncode == 0, result.stderr
    assert "could not install" in result.stderr


def _run_rocm_devel_headers_present(pythonpath: str, rocm_path: str) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("rocm_devel_headers_present")
    assert fn_src.strip(), "rocm_devel_headers_present() not found in install_baremetal.sh"
    script = f"set -uo pipefail\n{fn_src}\nrocm_devel_headers_present '{sys.executable}'\n"
    env = {"PYTHONPATH": pythonpath, "ROCM_PATH": rocm_path, "PATH": "/usr/bin:/bin"}
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True, env=env)


def test_rocm_devel_headers_present_finds_expanded_devel_wheel(tmp_path: Path) -> None:
    # Layout `rocm-sdk init` produces: headers under _rocm_sdk_devel/include.
    site = tmp_path / "site"
    hipblas = site / "_rocm_sdk_devel" / "include" / "hipblas"
    hipblas.mkdir(parents=True)
    (hipblas / "hipblas.h").write_text("")
    (site / "_rocm_sdk_devel" / "__init__.py").write_text("")

    result = _run_rocm_devel_headers_present(str(site), str(tmp_path / "no-rocm"))

    assert result.returncode == 0, result.stderr


def test_rocm_devel_headers_present_finds_standard_rocm_tree(tmp_path: Path) -> None:
    rocm = tmp_path / "opt-rocm"
    hipblas = rocm / "include" / "hipblas"
    hipblas.mkdir(parents=True)
    (hipblas / "hipblas.h").write_text("")
    empty_site = tmp_path / "empty"
    empty_site.mkdir()

    result = _run_rocm_devel_headers_present(str(empty_site), str(rocm))

    assert result.returncode == 0, result.stderr


def test_rocm_devel_headers_absent_on_runtime_only_wheel_stack(tmp_path: Path) -> None:
    # The failing rocm10 image: runtime packages installed, no devel headers.
    site = tmp_path / "site"
    (site / "_rocm_sdk_core" / "lib").mkdir(parents=True)
    (site / "_rocm_sdk_core" / "__init__.py").write_text("")
    (site / "_rocm_sdk_libraries" / "lib").mkdir(parents=True)
    (site / "_rocm_sdk_libraries" / "__init__.py").write_text("")

    result = _run_rocm_devel_headers_present(str(site), str(tmp_path / "no-rocm"))

    assert result.returncode == 1, result.stdout


def _run_ensure_rocm_devel_headers(
    tmp_path: Path,
    *,
    headers_present_initially: bool,
    core_version: str,
    pip_ok: bool = True,
    init_ok: bool = True,
    headers_present_after: bool = True,
) -> subprocess.CompletedProcess[str]:
    fn_src = _extract_function("ensure_rocm_devel_headers")
    assert fn_src.strip(), "ensure_rocm_devel_headers() not found in install_baremetal.sh"
    fake_py = tmp_path / "fake_python"
    fake_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-" ]; then cat >/dev/null; printf "%s" "$FAKE_CORE_VERSION"; exit 0; fi\n'
        'if [ "$2" = "pip" ]; then echo "PIP: $*"; exit "$FAKE_PIP_RC"; fi\n'
        'if [ "$2" = "rocm_sdk" ]; then echo "INIT: $*"; exit "$FAKE_INIT_RC"; fi\n'
        "exit 0\n"
    )
    fake_py.chmod(0o755)
    call_file = tmp_path / "present_calls"
    stub = f"""
log() {{ echo "LOG: $*"; }}
warn() {{ echo "WARN: $*" >&2; }}
ROCM_SDK_INDEX_URL="https://index.example/whl"
export FAKE_CORE_VERSION="{core_version}"
export FAKE_PIP_RC={0 if pip_ok else 1}
export FAKE_INIT_RC={0 if init_ok else 1}
rocm_devel_headers_present() {{
  _n=$(cat "{call_file}" 2>/dev/null || echo 0)
  echo $((_n + 1)) > "{call_file}"
  if [ "$_n" = "0" ]; then
    {"return 0" if headers_present_initially else "return 1"}
  fi
  {"return 0" if headers_present_after else "return 1"}
}}
"""
    script = f"set -uo pipefail\n{stub}\n{fn_src}\nensure_rocm_devel_headers '{fake_py}'\n"
    return subprocess.run(["bash", "-lc", script], check=False, capture_output=True, text=True)


def test_ensure_rocm_devel_headers_noop_when_headers_already_present(tmp_path: Path) -> None:
    result = _run_ensure_rocm_devel_headers(tmp_path, headers_present_initially=True, core_version="10.0.0")
    assert result.returncode == 0, result.stderr
    assert "PIP:" not in result.stdout
    assert "INIT:" not in result.stdout


def test_ensure_rocm_devel_headers_warns_when_rocm_is_not_wheel_based(tmp_path: Path) -> None:
    result = _run_ensure_rocm_devel_headers(tmp_path, headers_present_initially=False, core_version="")
    assert result.returncode == 0, result.stderr
    assert "not wheel-based" in result.stderr
    assert "PIP:" not in result.stdout


def test_ensure_rocm_devel_headers_pins_devel_to_installed_core_version(tmp_path: Path) -> None:
    result = _run_ensure_rocm_devel_headers(tmp_path, headers_present_initially=False, core_version="10.0.0")
    assert result.returncode == 0, result.stderr
    assert "rocm-sdk-devel==10.0.0" in result.stdout
    assert "--index-url https://index.example/whl" in result.stdout
    assert "INIT: -m rocm_sdk init" in result.stdout
    assert "ROCm devel headers ready" in result.stdout


def test_ensure_rocm_devel_headers_warns_and_continues_when_pip_fails(tmp_path: Path) -> None:
    result = _run_ensure_rocm_devel_headers(
        tmp_path, headers_present_initially=False, core_version="10.0.0", pip_ok=False
    )
    assert result.returncode == 0, result.stderr
    assert "could not install rocm-sdk-devel==10.0.0" in result.stderr
    assert "INIT:" not in result.stdout


def test_ensure_rocm_devel_headers_warns_and_continues_when_init_fails(tmp_path: Path) -> None:
    result = _run_ensure_rocm_devel_headers(
        tmp_path, headers_present_initially=False, core_version="10.0.0", init_ok=False
    )
    assert result.returncode == 0, result.stderr
    assert "rocm-sdk init failed" in result.stderr


def test_ensure_rocm_devel_headers_warns_when_headers_still_missing_after_init(tmp_path: Path) -> None:
    result = _run_ensure_rocm_devel_headers(
        tmp_path,
        headers_present_initially=False,
        core_version="10.0.0",
        headers_present_after=False,
    )
    assert result.returncode == 0, result.stderr
    assert "still not found" in result.stderr
