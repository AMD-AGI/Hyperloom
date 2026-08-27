"""Packaging resource smoke tests."""

from kernelforge.resources import resource_path


def test_packaged_resource_paths_are_available():
    assert (resource_path("knowledge_base") / "shared").is_dir()
    assert (resource_path("local_knowledge") / "hardware").is_dir()
    assert (resource_path("examples") / "flydsl-softmax-forge-loop").is_dir()


_VGPR_LIVENESS = (
    resource_path("local_knowledge")
    / "languages"
    / "asm"
    / "skills"
    / "optimize"
    / "asm_levers"
    / "intellikit"
    / "tools"
    / "scripts"
    / "vgpr_liveness.py"
)


def test_vgpr_liveness_no_re_instruction():
    assert "RE_INSTRUCTION" not in _VGPR_LIVENESS.read_text(encoding="utf-8"), (
        f"{_VGPR_LIVENESS}: RE_INSTRUCTION still present"
    )
