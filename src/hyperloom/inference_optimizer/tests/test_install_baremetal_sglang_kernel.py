# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sglang_kernel_layouts"
_INSTALL_SH = _ASSETS / "install_baremetal.sh"


def _kernel_dir(checkout: Path) -> subprocess.CompletedProcess[str]:
    fn_src = subprocess.run(
        ["sed", "-n", "/^sglang_kernel_rocm_build_dir()/,/^}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    script = f"set -euo pipefail\n{fn_src}\nsglang_kernel_rocm_build_dir '{checkout}'\n"
    return subprocess.run(
        ["bash", "-lc", script],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_suffix"),
    [
        ("legacy", "sgl-kernel"),
        ("v0517_aot", "python/sglang/kernels/aot"),
    ],
)
def test_sglang_kernel_rocm_build_dir_known_layouts(fixture_name: str, expected_suffix: str) -> None:
    checkout = _FIXTURES / fixture_name
    result = _kernel_dir(checkout)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith(expected_suffix)


def test_sglang_kernel_rocm_build_dir_missing_layout() -> None:
    checkout = _FIXTURES / "empty"
    result = _kernel_dir(checkout)
    assert result.returncode != 0


def test_sglang_reuse_checks_out_requested_ref_after_fallback_fetch(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(upstream)], check=True)
    tracked = upstream / "tracked"
    tracked.write_text("target\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(upstream), "add", "tracked"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "target",
        ],
        check=True,
    )
    target_ref = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("main\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qam",
            "main",
        ],
        check=True,
    )

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", str(upstream), str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "-q", target_ref], check=True)

    real_git = shutil.which("git")
    assert real_git is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git_wrapper = bin_dir / "git"
    git_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"fetch --depth 1 origin ${BLOCKED_REF}"* ]]; then exit 1; fi\n'
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)

    installer = _INSTALL_SH.read_text(encoding="utf-8")
    start_marker = '  else\n    git -C "$sglang_root" fetch --depth 1 origin "$SGLANG_REF"'
    start = installer.index(start_marker) + len("  else\n")
    end_marker = '    git -C "$sglang_root" submodule update --init --recursive\n'
    end = installer.index(end_marker, start) + len(end_marker)
    reuse_checkout = installer[start:end]
    env = {
        **os.environ,
        "BLOCKED_REF": target_ref,
        "REAL_GIT": real_git,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SGLANG_REF": target_ref,
        "sglang_root": str(checkout),
    }
    subprocess.run(["bash", "-euo", "pipefail", "-c", reuse_checkout], check=True, env=env)

    actual_ref = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert actual_ref == target_ref
