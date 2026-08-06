#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards for the opt-in pinned Origami installer."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
INSTALL_SH = REPO_ROOT / "src" / "hyperloom" / "agents" / "kernel" / "scripts" / "install.sh"


def _extract_function(name: str) -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", text, re.S | re.M)
    assert match, f"could not locate {name}() in install.sh"
    return match.group(0)


def _run_ensure_origami(
    tmp_path: Path,
    *,
    enabled: bool,
    dry_run: bool = False,
    check_only: bool = False,
    operator_root: bool = False,
    verify_ok: bool = True,
    source_present: bool = False,
) -> subprocess.CompletedProcess[str]:
    repo_root = tmp_path / "cache" / "rocm-libraries@abc"
    origami_root = (
        tmp_path / "operator" / "shared" / "origami"
        if operator_root
        else repo_root / "shared" / "origami"
    )
    if source_present:
        (origami_root / "python").mkdir(parents=True)
        (origami_root / "python" / "pyproject.toml").write_text(
            "[project]\nname='origami'\n",
            encoding="utf-8",
        )
    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log()  {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
die()  {{ echo "[die] $*" >&2; exit 1; }}
run()  {{ echo "RUN: $*"; }}
_resolve_ref_sha() {{ printf '%s' "$2"; }}
_project_name_from_pyproject() {{ echo origami; }}
_local_install_matches_root() {{ return 1; }}
_verify_origami_api() {{ return {0 if verify_ok else 1}; }}
CHECK_ONLY={int(check_only)}
DRY_RUN={int(dry_run)}
HYPERLOOM_ORIGAMI_GEMM_FALLBACK={1 if enabled else 0}
ORIGAMI_REPO=https://example.invalid/rocm-libraries.git
ORIGAMI_REF=abc
ORIGAMI_REPO_ROOT={repo_root}
ORIGAMI_ROOT={origami_root}
_origami_root_was_set={"1" if operator_root else '""'}
_open_source_root={tmp_path / "cache"}
ROCM_PATH=/opt/rocm

{_extract_function("_origami_install_enabled")}

{_extract_function("ensure_origami")}

ensure_origami
"""
    script = tmp_path / "origami-installer-harness.sh"
    script.write_text(harness, encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_disabled_is_zero_touch(tmp_path: Path) -> None:
    proc = _run_ensure_origami(tmp_path, enabled=False)
    assert proc.returncode == 0, proc.stdout
    assert "disabled; skipping" in proc.stdout
    assert "sparse clone" not in proc.stdout
    assert "pip install" not in proc.stdout
    assert not (tmp_path / "cache").exists()


def test_managed_dry_run_describes_sparse_clone_and_local_pip(tmp_path: Path) -> None:
    proc = _run_ensure_origami(tmp_path, enabled=True, dry_run=True)
    assert proc.returncode == 0, proc.stdout
    assert "sparse clone" in proc.stdout
    assert "sparse-checkout set shared/origami" in proc.stdout
    assert "/shared/origami/python" in proc.stdout
    assert not (tmp_path / "cache").exists()


def test_operator_root_dry_run_never_clones(tmp_path: Path) -> None:
    proc = _run_ensure_origami(
        tmp_path,
        enabled=True,
        dry_run=True,
        operator_root=True,
    )
    assert proc.returncode == 0, proc.stdout
    assert "validate operator Origami checkout" in proc.stdout
    assert "sparse clone" not in proc.stdout


def test_check_only_warns_without_mutation(tmp_path: Path) -> None:
    proc = _run_ensure_origami(
        tmp_path,
        enabled=True,
        check_only=True,
        verify_ok=False,
    )
    assert proc.returncode == 0, proc.stdout
    assert "Origami source missing" in proc.stdout
    assert "Python API is not importable" in proc.stdout
    assert not (tmp_path / "cache").exists()


def test_enabled_install_fails_when_api_verification_fails(tmp_path: Path) -> None:
    proc = _run_ensure_origami(
        tmp_path,
        enabled=True,
        operator_root=True,
        verify_ok=False,
        source_present=True,
    )
    assert proc.returncode != 0
    assert "required Python API verification failed" in proc.stdout


def test_static_pin_sparse_checkout_and_api_guards() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")
    body = _extract_function("ensure_origami")
    verify = _extract_function("_verify_origami_api")
    main_body = _extract_function("main")

    assert re.search(r'ORIGAMI_REF="\$\{ORIGAMI_REF:-[0-9a-f]{40}\}"', text)
    assert "git clone --filter=blob:none --no-checkout --sparse" in body
    assert "sparse-checkout set shared/origami" in body
    assert 'python3 -m pip install --no-cache-dir --break-system-packages "${root}/python"' in body
    assert "config_t" in verify
    assert "compute_total_latency" in verify
    assert "get_hardware_for_device" in verify
    protected = re.search(r"_DOTENV_PROTECTED_VARS='(.*?)'", text, re.S)
    assert protected
    for name in (
        "ORIGAMI_ROOT",
        "ORIGAMI_REPO",
        "ORIGAMI_REF",
        "HYPERLOOM_ORIGAMI_GEMM_FALLBACK",
    ):
        assert name in protected.group(1)
    assert main_body.index("ensure_origami") < main_body.index("ensure_tracelens")
