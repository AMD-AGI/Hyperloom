# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Guards for OOB source-path resolution in ``kernel-agent/scripts/install.sh``."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"

_BLOCK_RE = re.compile(
    r"# --- OOB source resolution \(BEGIN.*?\n(?P<body>.*?)\n# --- OOB source resolution \(END\) ---",
    re.S,
)


def _extract_block() -> str:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    m = _BLOCK_RE.search(text)
    assert m, "could not locate the OOB source resolution block in install.sh"
    return m.group("body")


def _resolve_oob_src(env: dict[str, str]) -> str:
    """Run the extracted resolution block in bash and return the resulting OOB_SRC."""
    block = _extract_block()
    # Provide the legacy-fallback variable the block references; the test sets
    # FORGE_PATH / OOB_SRC per-scenario via ``env``.
    script = (
        "set -euo pipefail\n"
        'HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/hyperloom}"\n'
        f"{block}\n"
        'printf "%s" "${OOB_SRC}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
    )
    assert proc.returncode == 0, f"resolution block failed: {proc.stderr}"
    return proc.stdout.strip()


def test_oob_src_derives_from_forge_path() -> None:
    """With FORGE_PATH set and no OOB_SRC, OOB resolves to $FORGE_PATH/OOB."""
    assert _resolve_oob_src({"FORGE_PATH": "/wekafs/KernelForge"}) == "/wekafs/KernelForge/OOB"


def test_oob_src_strips_trailing_slash_on_forge_path() -> None:
    assert _resolve_oob_src({"FORGE_PATH": "/wekafs/KernelForge/"}) == "/wekafs/KernelForge/OOB"


def test_oob_src_honours_forge_aliases() -> None:
    """KERNEL_FORGE_ROOT / KERNEL_FORGE_PATH are accepted like the forge backend."""
    assert _resolve_oob_src({"KERNEL_FORGE_ROOT": "/opt/KernelForge"}) == "/opt/KernelForge/OOB"
    assert _resolve_oob_src({"KERNEL_FORGE_PATH": "/opt/KF"}) == "/opt/KF/OOB"


def test_explicit_oob_src_is_honoured_over_forge_path() -> None:
    """An explicitly provided OOB_SRC always wins (operator override)."""
    out = _resolve_oob_src({"FORGE_PATH": "/wekafs/KernelForge", "OOB_SRC": "/custom/OOB"})
    assert out == "/custom/OOB"


def test_legacy_bundle_fallback_when_no_forge_path() -> None:
    """Without any forge path, fall back to the legacy $HYPERLOOM_BUNDLE/OOB layout."""
    assert _resolve_oob_src({"HYPERLOOM_BUNDLE": "/wekafs/hyperloom"}) == "/wekafs/hyperloom/OOB"


def test_install_script_no_longer_hardcodes_bundle_only_default() -> None:
    """Static guard: the OOB_SRC default must reference the forge path."""
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert '_forge_root="${FORGE_PATH:-${KERNEL_FORGE_ROOT:-${KERNEL_FORGE_PATH:-}}}"' in text
    assert 'OOB_SRC="${OOB_SRC:-${_forge_root%/}/OOB}"' in text
