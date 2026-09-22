# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior tests for bare-metal vLLM source and wheel routing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_INSTALL_SH = Path(__file__).resolve().parents[1] / "assets" / "install_baremetal.sh"
_REF = "98dff2a81d747d1dba01a47f939f48c3526d4206"


def _functions(*names: str) -> str:
    text = _INSTALL_SH.read_text()
    chunks = []
    for name in names:
        result = subprocess.run(
            ["sed", "-n", f"/^{name}()/,/^}}/p", str(_INSTALL_SH)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert result.strip(), f"{name}() not found in install_baremetal.sh"
        chunks.append(result)
    return "\n".join(chunks)


def _bash(functions: tuple[str, ...], body: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    script = f"set -uo pipefail\n{_functions(*functions)}\n{body}\n"
    return subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True, env=env)


@pytest.mark.parametrize(
    ("release", "hip", "configured", "expected", "ok"),
    [
        ("10.0", "7.15.0", "auto", "source", True),
        ("7.2", "7.2.4", "auto", "wheel", True),
        ("", "7.2", "auto", "wheel", True),
        ("", "7.2.4", "wheel", "wheel", True),
        ("10.0", "7.15.0", "wheel", "", False),
        ("", "7.15.0", "auto", "", False),
        ("10.0", "7.2.4", "auto", "", False),
        ("", "", "auto", "", False),
    ],
)
def test_route_matrix(release: str, hip: str, configured: str, expected: str, ok: bool) -> None:
    result = _bash(
        ("vllm_install_method_for_stack",),
        f'VLLM_INSTALL_METHOD="{configured}"; vllm_install_method_for_stack "{release}" "{hip}"',
    )
    assert (result.returncode == 0) is ok
    assert result.stdout.strip() == expected


def test_unknown_stack_fails_before_mutation(tmp_path: Path) -> None:
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\ncat >/dev/null; echo 3.12\n")
    fake_py.chmod(0o755)
    body = f"""
log() {{ :; }}; warn() {{ :; }}; die() {{ echo "$*" >&2; return 1; }}
resolve_python() {{ echo "{fake_py}"; }}
route_vllm_install_method() {{ die "unsupported stack"; }}
ensure_openmpi_runtime() {{ echo MUTATION; }}
FRAMEWORK_ENV=isolated; VLLM_VENV_ROOT="{tmp_path}/venv"; CHECK_ONLY=0; DRY_RUN=0
VLLM_VERSION=0.29.0; VLLM_ROCM_VARIANT=rocm723; VLLM_ROCM_INDEX=https://example.invalid
install_vllm_framework
"""
    result = _bash(("install_vllm_framework",), body)
    assert result.returncode != 0
    assert "MUTATION" not in result.stdout


@pytest.mark.parametrize("mode", ["dry", "check"])
def test_source_non_mutating_modes(tmp_path: Path, mode: str) -> None:
    dry, check = (1, 0) if mode == "dry" else (0, 1)
    body = f"""
log() {{ echo "$*"; }}; warn() {{ echo "$*" >&2; }}; die() {{ echo "$*" >&2; return 1; }}
rocm_devel_headers_present() {{ return 1; }}
ensure_rocm_devel_headers() {{ echo MUTATION; }}
check_vllm_source_prereqs() {{ echo MUTATION; }}
ensure_vllm_checkout() {{ echo MUTATION; }}
verify_vllm_source() {{ return 1; }}
DRY_RUN={dry}; CHECK_ONLY={check}; VLLM_SOURCE_REF={_REF}
VLLM_ROOT="{tmp_path}/src"; VLLM_VENV_ROOT="{tmp_path}/venv"
install_vllm_from_source /usr/bin/python3
"""
    result = _bash(("install_vllm_from_source",), body)
    assert result.returncode == 0, result.stderr
    assert "MUTATION" not in result.stdout
    assert ("would ensure ROCm devel headers" if mode == "dry" else "not installed") in (result.stdout + result.stderr)


@pytest.mark.parametrize("case", ["dirty", "origin"])
def test_checkout_rejects_unsafe_existing_tree(tmp_path: Path, case: str) -> None:
    root = tmp_path / "vllm"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    (root / "tracked").write_text("clean")
    subprocess.run(["git", "-C", str(root), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    repo = "https://example.invalid/vllm.git"
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", repo], check=True)
    if case == "dirty":
        (root / "tracked").write_text("dirty")
    else:
        repo = "https://other.invalid/vllm.git"
    body = f"""
die() {{ echo "$*" >&2; exit 1; }}
VLLM_REPO="{repo}"; VLLM_SOURCE_REF={_REF}
ensure_vllm_checkout "{root}"
"""
    result = _bash(("vllm_checkout_state", "ensure_vllm_checkout"), body)
    assert result.returncode != 0
    assert case in result.stderr.lower()


def _source_stubs(tmp_path: Path, checkout: str) -> str:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake_py = venv / "bin" / "python"
    fake_py.write_text(
        '#!/bin/sh\necho "PY:$* CONSTRAINT=${PIP_CONSTRAINT:-} TARGET=${VLLM_TARGET_DEVICE:-}" >> "$CALLS"\n'
    )
    fake_py.chmod(0o755)
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    return f"""
log() {{ echo "$*"; }}; warn() {{ echo "$*" >&2; }}; die() {{ echo "$*" >&2; return 1; }}
ensure_vllm_checkout() {{ echo {checkout}; }}
vllm_overlay_is_valid() {{ return 0; }}
verify_vllm_source() {{ return 0; }}
check_vllm_source_prereqs() {{ echo PREREQS >> "$CALLS"; }}
check_vllm_source_python() {{ :; }}
rocm_devel_headers_present() {{ return 0; }}
ensure_rocm_devel_headers() {{ echo HEADERS >> "$CALLS"; }}
detect_rocm_gfx_arch() {{ echo gfx950; }}
write_rocm_torch_constraints() {{ echo torch==2.12.0 > "$2"; echo triton==3.6.0 >> "$2"; }}
inherit_vllm_base_site_packages() {{ echo BASE_SITE >> "$CALLS"; }}
link_vllm_into_shared_bin() {{ echo LINK >> "$CALLS"; }}
export CALLS="{tmp_path}/calls"; DRY_RUN=0; CHECK_ONLY=0; VLLM_SOURCE_REF={_REF}
VLLM_ROOT="{tmp_path}/src"; VLLM_VENV_ROOT="{venv}"; VLLM_REPO=https://example.invalid/vllm.git
ROCM_SDK_INDEX_URL=https://stable.repo.amd.com/rocm/whl-next
mkdir -p "$VLLM_ROOT/requirements"; : > "$VLLM_ROOT/requirements/rocm.txt"
"""


def test_source_exact_checkout_fast_path_skips_build(tmp_path: Path) -> None:
    result = _bash(
        ("install_vllm_from_source",),
        _source_stubs(tmp_path, "exact") + f'\ninstall_vllm_from_source "{tmp_path}/venv/bin/python"\n',
        env={**os.environ},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls").read_text() == "BASE_SITE\nLINK\n"


def test_source_build_preserves_constraint_for_develop(tmp_path: Path) -> None:
    result = _bash(
        ("install_vllm_from_source",),
        _source_stubs(tmp_path, "updated") + f'\ninstall_vllm_from_source "{tmp_path}/venv/bin/python"\n',
        env={**os.environ},
    )
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text()
    assert calls.index("BASE_SITE") < calls.index("pip install")
    assert "pip install --no-build-isolation --constraint" in calls
    assert "--extra-index-url https://stable.repo.amd.com/rocm/whl-next" in calls
    assert "setup.py develop --no-deps CONSTRAINT=/tmp/" in calls
    assert "TARGET=rocm" in calls


def test_source_route_requires_isolated_framework_env(tmp_path: Path) -> None:
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\nexit 0\n")
    fake_py.chmod(0o755)
    body = f"""
die() {{ echo "$*" >&2; return 1; }}
resolve_python() {{ echo "{fake_py}"; }}
route_vllm_install_method() {{ VLLM_INSTALL_METHOD=source; }}
install_vllm_from_source() {{ echo MUTATION; }}
FRAMEWORK_ENV=shared
install_vllm_framework
"""
    result = _bash(("install_vllm_framework",), body)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "requires --framework-env isolated" in result.stderr


def test_overlay_inherits_sdk_amdsmi_path(tmp_path: Path) -> None:
    base_site = tmp_path / "base-site"
    overlay_site = tmp_path / "overlay-site"
    amdsmi = base_site / "_rocm_sdk_core" / "share" / "amd_smi"
    amdsmi.mkdir(parents=True)
    overlay_site.mkdir()
    base_py = tmp_path / "base-python"
    overlay_py = tmp_path / "overlay-python"
    base_py.write_text(f"#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{base_site}'\n")
    overlay_py.write_text(f"#!/bin/sh\ncat >/dev/null\nprintf '%s\\n' '{overlay_site}'\n")
    base_py.chmod(0o755)
    overlay_py.chmod(0o755)
    body = f"""
die() {{ echo "$*" >&2; exit 1; }}
inherit_vllm_base_site_packages "{base_py}" "{overlay_py}"
"""
    result = _bash(("inherit_vllm_base_site_packages",), body)
    assert result.returncode == 0, result.stderr
    inherited = (overlay_site / "hyperloom-base-venv.pth").read_text().splitlines()
    assert inherited == [str(base_site), str(amdsmi)]


def test_source_roots_are_persisted() -> None:
    body = """
resolve_installed_framework() { echo vllm; }; log() { :; }; warn() { :; }
upsert_dotenv_var() { echo "$1=$2"; }; remove_dotenv_var() { :; }
DRY_RUN=0; CHECK_ONLY=0; DOTENV=/tmp/.env; USER_DATA_PATH=/data; FRAMEWORK_ENV=isolated; INSTALL_FRAMEWORK=vllm
VLLM_INSTALL_METHOD=source; VLLM_ROOT=/src/vllm; FRAMEWORK_REPO_PATH=/src/vllm; VLLM_VENV_ROOT=/venv
VLLM_TARGET_DEVICE=rocm
PYTHON=; INFERENCE_OPTIMIZER_FORCE_PYTHON=; VIRTUAL_ENV=; ROCM_PATH=; HIP_PATH=
SGLANG_ROCM_EXTRA=; SGLANG_ROCM_PYPI_VERSION=; AITER_REF=; KERNEL_OPT_BACKEND_ORDER=
HYPERLOOM_WHEEL_REPO=; HYPERLOOM_WHEEL_TAG=; HYPERLOOM_SKILL_PATH=; SGLANG_USE_AITER=
write_runtime_dotenv
"""
    result = _bash(("write_runtime_dotenv",), body)
    assert "VLLM_ROOT=/src/vllm" in result.stdout
    assert "FRAMEWORK_REPO_PATH=/src/vllm" in result.stdout
    assert "VLLM_TARGET_DEVICE=rocm" in result.stdout


def test_wheel_torch_index_resolution_failure_is_fatal(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text('#!/bin/sh\nif [ "$1" = "-" ]; then cat >/dev/null; [ "$#" -eq 1 ] && echo 3.12; fi\n')
    py.chmod(0o755)
    body = f"""
log() {{ :; }}; warn() {{ :; }}; die() {{ echo "$*" >&2; exit 1; }}
resolve_python() {{ echo "{py}"; }}; route_vllm_install_method() {{ VLLM_INSTALL_METHOD=wheel; }}
ensure_openmpi_runtime() {{ :; }}; assert_vllm_glibc_compatible() {{ :; }}
verify_vllm_rocm() {{ return 0; }}; link_vllm_into_shared_bin() {{ :; }}
FRAMEWORK_ENV=isolated; VLLM_VENV_ROOT="{venv}"; CHECK_ONLY=0; DRY_RUN=0
VLLM_VERSION=0.29.0; VLLM_ROCM_VARIANT=rocm723; VLLM_ROCM_INDEX=https://example.invalid
install_vllm_framework
"""
    result = _bash(("install_vllm_framework",), body)
    assert result.returncode != 0
    assert "resolve" in result.stderr.lower()
