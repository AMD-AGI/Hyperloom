# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installer ownership of existing and newly cloned InferenceX checkouts."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


INSTALL_SCRIPT = Path(__file__).resolve().parents[1] / "assets" / "install.sh"


def _run_inferencex_install(tmp_path: Path, *, explicit_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^ensure_inferencex\(\) \{.*?^\}", text, re.S | re.M)
    assert match is not None
    script = tmp_path / "installer-harness.sh"
    script.write_text(
        """set -euo pipefail
log() { :; }
warn() { echo "$*" >&2; }
CHECK_ONLY=0
DRY_RUN=0
INFERENCEX_REPO=https://example.invalid/InferenceX.git
INFERENCEX_REF=deadbeef
git_fetch_pinned() { mkdir -p "$2/benchmarks"; }
ensure_inferencex_aiperf_submodule() {
  printf '%s\\n' "$INFERENCEX_PATH" >> "$SUBMODULE_UPDATE_LOG"
}
"""
        + match.group(0)
        + '\nensure_inferencex\nprintf "%s\\n" "$INFERENCEX_PATH"\n',
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "INFERENCEX_PATH": str(explicit_path or ""),
            "INFERENCEX_DEFAULT_DIR": str(tmp_path / "cached"),
            "SUBMODULE_UPDATE_LOG": str(tmp_path / "submodule-updates"),
        },
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("has_aiperf", [True, False])
def test_installer_preserves_existing_checkout_without_submodule_updates(tmp_path, explicit, has_aiperf):
    checkout = tmp_path / ("user-checkout" if explicit else "cached")
    (checkout / "benchmarks").mkdir(parents=True)
    launcher = checkout / "benchmarks" / "custom.sh"
    launcher.write_text("# operator-owned launcher\n", encoding="utf-8")
    if has_aiperf:
        submodule = checkout / "utils" / "aiperf"
        submodule.mkdir(parents=True)
        (submodule / "pyproject.toml").write_text("# operator-owned revision\n", encoding="utf-8")
    before = {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob("*") if path.is_file()}

    result = _run_inferencex_install(tmp_path, explicit_path=checkout if explicit else None)

    assert result.stdout.strip() == str(checkout)
    assert not (tmp_path / "submodule-updates").exists()
    assert {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob("*") if path.is_file()} == before


def test_installer_initializes_submodule_for_its_new_clone(tmp_path):
    result = _run_inferencex_install(tmp_path)

    assert result.stdout.strip() == str(tmp_path / "cached")
    assert (tmp_path / "submodule-updates").read_text().splitlines() == [str(tmp_path / "cached")]
