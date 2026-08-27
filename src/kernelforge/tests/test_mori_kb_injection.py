"""Integration coverage for the MoRI knowledge-base injection knob.

``include_mori_kb`` is an experimental, off-by-default ablation knob (see
``Config.include_mori_kb`` and ``build_forge_knowledge(include_mori=...)``).
These tests close the gap flagged in review: nothing previously exercised
the default-off behavior, env-var-enabled behavior, the ``build_
forge_knowledge`` flag directly, ``Config.from_env`` forwarding, explicit-
False-vs-env-var precedence, or wheel packaging of ``framework/mori/``.
"""

from __future__ import annotations

import re
from pathlib import Path

from kernelforge.config import Config
from kernelforge.fellows.base import build_single_fellow_prompt
from kernelforge.knowledge.local_index import build_forge_knowledge
from kernelforge.resources import resource_path


def _wheel_force_include() -> dict[str, str]:
    """``[tool.hatch.build.targets.wheel.force-include]``, via TOML parse or a raw-text
    fallback on py3.10 CI runners that ship neither ``tomllib`` (3.11+) nor ``tomli``."""
    raw = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            tomllib = None
    if tomllib is not None:
        pyproject = tomllib.loads(raw)
        return pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    section = re.search(
        r"\[tool\.hatch\.build\.targets\.wheel\.force-include\]\n(.*?)(?:\n\[|\Z)",
        raw,
        re.DOTALL,
    )
    assert section, "no [tool.hatch.build.targets.wheel.force-include] section found"
    return dict(re.findall(r'"([^"]+)"\s*=\s*"([^"]+)"', section.group(1)))


def _mori_prompt(config: Config) -> str:
    """A prompt from a fellow whose backend actually has a framework/mori/ folder to inject."""
    return build_single_fellow_prompt(
        config,
        "aiter-fellow",
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
    # Regression test: from_env() previously never forwarded this kwarg at
    # all, so an explicit override was silently dropped in favor of the env
    # var (or the False default) every time.
    monkeypatch.delenv("KERNELFORGE_INCLUDE_MORI_KB", raising=False)
    config = Config.from_env(include_mori_kb=True)
    assert config.include_mori_kb is True

    monkeypatch.setenv("KERNELFORGE_INCLUDE_MORI_KB", "1")
    config = Config.from_env(include_mori_kb=False)
    assert config.include_mori_kb is False


def test_explicit_false_beats_env_var(monkeypatch):
    # Regression test: include_mori_kb used to be a plain bool defaulting to
    # False, so __post_init__ couldn't distinguish "explicitly False" from
    # "not specified" and always re-derived from the env var whenever falsy.
    monkeypatch.setenv("KERNELFORGE_INCLUDE_MORI_KB", "1")
    config = Config(gpu_target="gfx942", include_mori_kb=False)
    assert config.include_mori_kb is False


def test_wheel_includes_mori_kb():
    force_include = _wheel_force_include()
    assert force_include.get("local_knowledge") == "kernelforge/data/local_knowledge"

    mori_kb_dir = resource_path("local_knowledge") / "framework" / "mori"
    assert mori_kb_dir.is_dir()
    assert any(mori_kb_dir.rglob("*.md")), "framework/mori/ has no .md cards to ship"
