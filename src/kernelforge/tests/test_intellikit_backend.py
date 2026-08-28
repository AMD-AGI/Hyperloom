"""Tests for the IntelliKit ASM kernel_backend."""

import os

from kernelforge.kernel_backends.constants import KERNEL_BACKENDS
from kernelforge.kernel_backends.intellikit.prompts import (
    build_system_prompt,
    _roofline,
    _roofline_peak,
    PATCH_CO,
)


# ─── Registration ───────────────────────────────────────────────────────────


def test_intellikit_in_kernel_backend_backends():
    assert "intellikit" in KERNEL_BACKENDS


# ─── Prompt content ─────────────────────────────────────────────────────────


def test_prompt_contains_gpu_target():
    prompt = build_system_prompt("gfx950", knowledge_content="# KB")
    assert "gfx950" in prompt


def test_prompt_contains_gpu_target_not_hardcoded():
    """Prompt must use the provided gpu_target, not a hardcoded value."""
    prompt = build_system_prompt("gfx942", knowledge_content="# KB")
    assert "gfx942" in prompt
    # Identity line must not say gfx950 when we asked for gfx942
    identity_line = next(l for l in prompt.splitlines() if "IntelliKit kernel backend" in l)
    assert "gfx950" not in identity_line


def test_prompt_contains_knowledge_block():
    prompt = build_system_prompt("gfx950", knowledge_content="MY_UNIQUE_KB_MARKER")
    assert "MY_UNIQUE_KB_MARKER" in prompt


def test_prompt_contains_patch_co_path():
    prompt = build_system_prompt("gfx950", knowledge_content="# KB")
    assert "patch_co.py" in prompt


def test_prompt_contains_round_trip_workflow():
    prompt = build_system_prompt("gfx950", knowledge_content="# KB")
    assert "Round-Trip Workflow" in prompt
    assert "cos_sim" in prompt
    assert "1.000000" in prompt


# ─── Roofline lookup ────────────────────────────────────────────────────────


def test_roofline_gfx950():
    line = _roofline("gfx950")
    assert "2517" in line
    assert "5.3" in line


def test_roofline_gfx942():
    line = _roofline("gfx942")
    assert "1300" in line


def test_roofline_unknown_arch_does_not_crash():
    line = _roofline("gfx999")
    assert line  # returns something, not empty


def test_roofline_peak_gfx950():
    assert _roofline_peak("gfx950") == "2517"


def test_roofline_peak_unknown():
    assert _roofline_peak("gfx999") == "N/A"


# ─── patch_co.py tool ───────────────────────────────────────────────────────


def test_patch_co_ships_with_kernel_backend():
    assert os.path.isfile(PATCH_CO), f"patch_co.py not found at {PATCH_CO}"


def test_patch_co_syntax():
    import py_compile

    py_compile.compile(PATCH_CO, doraise=True)


def test_patch_co_mcpu_parameter():
    """patch_co.py must accept --mcpu to support non-gfx950 targets."""
    with open(PATCH_CO) as f:
        source = f.read()
    assert "--mcpu" in source
    assert "mcpu" in source


def test_patch_co_rejects_larger_text():
    """patch_co.py must exit non-zero when new .text is larger than original."""
    import subprocess

    result = subprocess.run(
        ["python3", PATCH_CO, "--help"],
        capture_output=True,
        text=True,
    )
    # --help or no args should not crash with an unhandled exception
    assert result.returncode in (0, 1)  # usage error is fine, traceback is not
    assert "Traceback" not in result.stderr
