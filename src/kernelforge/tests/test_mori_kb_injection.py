"""Integration coverage for the MoRI knowledge-base injection knob."""

from __future__ import annotations


from kernelforge.config import Config
from kernelforge.kernel_backends.base import build_single_kernel_backend_prompt
from kernelforge.knowledge.local_index import build_forge_knowledge
from kernelforge.resources import resource_path


def _mori_prompt(config: Config) -> str:
    """A prompt from a kernel backend whose backend actually has a framework/mori/ folder to inject."""
    return build_single_kernel_backend_prompt(
        config,
        "aiter",
        task_type="repository",
        source_paths=["/work/mori_ep_dispatch_combine/driver.py"],
    )


def test_default_no_mori_kb_injection(monkeypatch):
    monkeypatch.delenv("KERNELFORGE_INCLUDE_MORI_KB", raising=False)
    config = Config(gpu_target="gfx942")
    assert config.include_mori_kb is False
    assert "framework/mori" not in _mori_prompt(config)


def test_env_var_enables_mori_kb_injection(monkeypatch):
    monkeypatch.setenv("KERNELFORGE_INCLUDE_MORI_KB", "1")
    config = Config(gpu_target="gfx942")
    assert config.include_mori_kb is True
    assert "framework/mori" in _mori_prompt(config)


def test_build_forge_knowledge_include_mori_flag():
    root = resource_path("local_knowledge")
    with_mori = build_forge_knowledge(root, include_mori=True)
    without_mori = build_forge_knowledge(root, include_mori=False)
    assert "framework/mori" in with_mori
    assert "framework/mori" not in without_mori


def test_config_from_env_include_mori_kb_override(monkeypatch):
    # Regression test: from_env() previously never forwarded this kwarg at all, so an explicit override was silently
    # dropped in favor of the env var (or the False default) every time.
    monkeypatch.delenv("KERNELFORGE_INCLUDE_MORI_KB", raising=False)
    config = Config.from_env(include_mori_kb=True)
    assert config.include_mori_kb is True

    monkeypatch.setenv("KERNELFORGE_INCLUDE_MORI_KB", "1")
    config = Config.from_env(include_mori_kb=False)
    assert config.include_mori_kb is False


def test_explicit_false_beats_env_var(monkeypatch):
    # Regression test: include_mori_kb used to be a plain bool defaulting to False, so __post_init__ couldn't
    # distinguish "explicitly False" from "not specified" and always re-derived from the env var whenever falsy.
    monkeypatch.setenv("KERNELFORGE_INCLUDE_MORI_KB", "1")
    config = Config(gpu_target="gfx942", include_mori_kb=False)
    assert config.include_mori_kb is False


def test_mori_kb_ships_with_the_package():
    """The MoRI cards must resolve through the packaged data tree."""
    mori_kb_dir = resource_path("local_knowledge") / "framework" / "mori"
    assert mori_kb_dir.is_dir()
    assert any(mori_kb_dir.rglob("*.md")), "framework/mori/ has no .md cards to ship"
