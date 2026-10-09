# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavioural + static guards for ``ensure_magpie()``."""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
PREFLIGHT = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "cli" / "preflight.py"

MAGPIE_AGENTX_COMMIT = "d80eb4d3dad7fabe01ce81d049e2983adf2c86dd"

PIP_MARKER = "pip-install-called"


def _extract_ensure_magpie() -> str:
    text = IO_INSTALL.read_text(encoding="utf-8")
    m = re.search(r"^ensure_magpie\(\) \{.*?^\}", text, re.S | re.M)
    assert m, "could not locate ensure_magpie() in install.sh"
    return m.group(0)


def _fake_python(tmp_path: Path, *, capability_ok: bool, installed_root: Path, repair_required: bool = False) -> Path:
    """A stub ``$PYTHON``."""
    marker = tmp_path / PIP_MARKER
    repaired = tmp_path / "pip-force-reinstall-called"
    health_marker = repaired if repair_required else marker
    capability_check = "exit 0" if capability_ok else f'[ -f "{health_marker}" ] && exit 0 || exit 1'
    body = f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  printf '%s\n' "$*" >> "{marker}"
  case " $* " in
    *" --force-reinstall "*) touch "{repaired}" ;;
  esac
  exit 0
fi
if [ "$1" = "-" ]; then
  echo "{installed_root}"
  exit 0
fi
if [ "$1" = "-c" ]; then
  {capability_check}
fi
exit 0
"""
    py = tmp_path / "fake_python.sh"
    py.write_text(body, encoding="utf-8")
    py.chmod(py.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return py


def _run_ensure_magpie(
    tmp_path: Path,
    *,
    capability_ok: bool,
    explicit_magpie_path: bool = False,
    repair_required: bool = False,
) -> tuple[str, bool]:
    """Run the extracted ensure_magpie body; return (stdout, pip_called)."""
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir(parents=True)
    magpie_dir = tmp_path / "operator" / "Magpie"
    fake_py = _fake_python(
        tmp_path, capability_ok=capability_ok, installed_root=installed_root, repair_required=repair_required
    )
    magpie_path_line = (
        f'MAGPIE_PATH="{magpie_dir}"\nMAGPIE_PATH_EXPLICIT=1'
        if explicit_magpie_path
        else f'MAGPIE_PATH="{magpie_dir}"\nMAGPIE_PATH_EXPLICIT=0'
    )

    harness = f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[log] $*"; }}
warn() {{ echo "[warn] $*"; }}
CHECK_ONLY=0
DRY_RUN=0
MAGPIE_REPO="https://example.invalid/Magpie.git"
MAGPIE_REF="deadbeef"
MAGPIE_PACKAGE_SPEC="magpie-eval @ git+https://example.invalid/Magpie.git@deadbeef"
{magpie_path_line}
PYTHON="{fake_py}"
PIP_EXTRA=("--disable-pip-version-check")

{_extract_ensure_magpie()}

ensure_magpie
"""
    script = tmp_path / "harness.sh"
    script.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, f"ensure_magpie harness failed:\n{proc.stdout}"
    pip_called = (tmp_path / PIP_MARKER).exists()
    return proc.stdout, pip_called


def test_skip_reinstall_when_native_agentx_capability_is_available(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=True)
    assert not pip_called, f"reinstall should have been skipped:\n{out}"
    assert "Magpie package healthy; skipping pip install" in out
    assert "MAGPIE_PATH resolved from installed package" in out


def test_install_when_native_agentx_capability_is_missing(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=False)
    assert pip_called, f"reinstall should have run on capability miss:\n{out}"
    assert "Magpie installed OK from magpie-eval @ git+https://example.invalid/Magpie.git@deadbeef" in out


def test_repairs_an_installed_commit_whose_execution_tree_was_patched(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=False, repair_required=True)

    assert pip_called
    calls = (tmp_path / PIP_MARKER).read_text().splitlines()
    assert len(calls) == 2
    assert "--force-reinstall" not in calls[0]
    assert "--force-reinstall --no-deps" in calls[1]
    assert "Magpie installed OK" in out


def test_preserves_explicit_magpie_path(tmp_path: Path) -> None:
    out, pip_called = _run_ensure_magpie(tmp_path, capability_ok=True, explicit_magpie_path=True)
    assert not pip_called
    assert "MAGPIE_PATH override preserved" in out


# Static guard: the idempotent skip must stay wired in.
def test_io_install_magpie_reinstall_is_idempotent_guarded() -> None:
    body = _extract_ensure_magpie()
    assert "magpie_health_code(native_agentx=native_agentx_session())" in body
    assert "Magpie package healthy; skipping pip install" in body
    assert "MAGPIE_PACKAGE_SPEC" in body
    assert "pip install" in body, "reinstall path must still exist for the miss case"


def test_default_magpie_pin_has_native_agentx_and_is_consistent() -> None:
    install_text = IO_INSTALL.read_text(encoding="utf-8")
    preflight_text = PREFLIGHT.read_text(encoding="utf-8")

    assert f'MAGPIE_REF="${{MAGPIE_REF:-{MAGPIE_AGENTX_COMMIT}}}"' in install_text
    assert f'_MAGPIE_REF_DEFAULT = "{MAGPIE_AGENTX_COMMIT}"' in preflight_text
    assert 'os.environ.get("MAGPIE_REF") or _MAGPIE_REF_DEFAULT' in preflight_text


@pytest.mark.parametrize(
    "mode, requires_native",
    [
        ("synthetic", False),
        ("native-switch", True),
        ("native-yaml", True),
        ("legacy-resume", False),
        ("native-resume", True),
        ("mlperf", False),
    ],
)
def test_installer_checks_the_session_contract_for_an_importable_legacy_package(tmp_path, mode, requires_native):
    import json
    import os
    import sys

    package = tmp_path / "packages" / "Magpie"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# Importable Magpie without AgentX support.\n")
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1:3] == ['-m', 'pip']:\n"
        "    print('pip-install-called'); sys.exit(71)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
    )
    python.chmod(0o755)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(package.parent), str(REPO_ROOT / "src")]),
        HYPERLOOM_AGENTX="1" if mode in {"native-switch", "legacy-resume", "mlperf"} else "0",
        HYPERLOOM_AGENTIC_BACKEND="mlperf" if mode == "mlperf" else "aiperf",
        HYPERLOOM_BENCHMARK_CONFIG="",
        INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR="",
    )
    if mode == "native-yaml":
        config = tmp_path / "native.yaml"
        config.write_text("benchmark:\n  agentx: enable\n")
        env["HYPERLOOM_BENCHMARK_CONFIG"] = str(config)
    if mode.endswith("resume"):
        (tmp_path / "state.json").write_text(
            json.dumps({"benchmark_mode": "agentx", "agentx_epoch": 1 if mode == "legacy-resume" else 4})
        )
        env["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(tmp_path)
    script = tmp_path / "install.sh"
    script.write_text(
        'set -euo pipefail\nlog() { echo "$*"; }\nwarn() { echo "$*"; }\n'
        f'PYTHON="{python}"\nMAGPIE_REF=custom-legacy-ref\nMAGPIE_PACKAGE_SPEC=local-package\n'
        "CHECK_ONLY=0\nDRY_RUN=0\nMAGPIE_PATH_EXPLICIT=0\nPIP_EXTRA=()\n"
        + _extract_ensure_magpie()
        + "\nensure_magpie\n"
    )
    proc = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)
    assert proc.returncode == (71 if requires_native else 0), proc.stderr
    assert (PIP_MARKER in proc.stdout) is requires_native
    if not requires_native:
        assert "skipping pip install" in proc.stdout
        assert str(package.parent) in proc.stdout
