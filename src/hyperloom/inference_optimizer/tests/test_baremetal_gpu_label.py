# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``detect_gpu_label`` must never invent an MI part number it did not probe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import setup


def _gpu_label_function() -> str:
    """Return just the ``detect_gpu_label`` definition from the installer."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("detect_gpu_label() {")
    end = script_text.index("\nDETECTED_GPU=")
    return script_text[start:end]


def _stub_rocm_smi(tmp_path: Path, product_name: str) -> Path:
    """Stub rocm-smi so the probe is deterministic on any host.

    CI runners have no ROCm and dev boxes may hold any GPU, so the real binary
    would make these assertions machine-dependent.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir(exist_ok=True)
    rocm_smi = stub_bin / "rocm-smi"
    rocm_smi.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "{product_name}"\n',
        encoding="utf-8",
    )
    rocm_smi.chmod(0o755)
    return stub_bin


def _detect_gpu_label(
    tmp_path: Path,
    gfx: str,
    product_name: str = "AMD Radeon 8060S Graphics",
    strict: bool = False,
) -> str:
    """Invoke the installer's ``detect_gpu_label`` and return its label.

    ``strict`` mirrors the installer's own ``set -euo pipefail`` while calling
    the function directly, which is the arrangement that a no-match ``grep``
    can abort.
    """
    stub_bin = _stub_rocm_smi(tmp_path, product_name)
    out_file = tmp_path / "label.txt"
    runner = tmp_path / "run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail" if strict else "set -uo pipefail",
                f'PATH="{stub_bin}:$PATH"',
                _gpu_label_function(),
                f'detect_gpu_label "{gfx}" > {out_file}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)
    return out_file.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(
    ("gfx", "expected"),
    [
        ("gfx942", "MI300X"),
        ("gfx950", "MI355X"),
        # Unsupported/untested arches report what was probed, not a stand-in.
        # gfx1151 (Strix Halo) previously reported MI300X.
        ("gfx1151", "gfx1151"),
        ("gfx90a", "gfx90a"),
        # Nothing probed at all.
        ("", "unknown"),
    ],
)
def test_detect_gpu_label_reports_probed_arch(tmp_path: Path, gfx: str, expected: str):
    assert _detect_gpu_label(tmp_path, gfx) == expected


def test_detect_gpu_label_never_fabricates_mi_part_for_unknown_arch(tmp_path: Path):
    """Regression guard for the specific failure mode.

    The label is substituted into the operator prompt as ``GPU: <label>``, so an
    invented MI300X propagates into ``--gpu-type`` and mis-selects the Magpie
    runner scripts.
    """
    assert "MI" not in _detect_gpu_label(tmp_path, "gfx1151").upper()


def test_detect_gpu_label_prefers_rocm_smi_product_name(tmp_path: Path):
    """A real MI product name still wins over the arch table."""
    label = _detect_gpu_label(tmp_path, "gfx942", product_name="Card series: AMD Instinct MI325X")
    assert label == "MI325X"


def test_detect_gpu_label_survives_no_match_grep_under_set_e(tmp_path: Path):
    """A non-MI product name must not trip ``pipefail``/``set -e``."""
    assert _detect_gpu_label(tmp_path, "gfx1151", strict=True) == "gfx1151"
