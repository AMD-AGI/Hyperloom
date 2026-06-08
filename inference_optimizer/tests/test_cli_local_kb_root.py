# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the recipe-snapshot KB dispatcher bootstrap helpers in cli."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from inference_optimizer.cli import (
    _build_recipe_kb_dispatcher,
    _resolve_local_kb_root,
)
from inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    RemoteRecipeClient,
)


# Fixtures
@pytest.fixture
def env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe the env vars these helpers consult so each test's precedence tier is explicit."""
    for key in (
        "HYPERLOOM_LOCAL_KB_ROOT",
        "USER_DATA_PATH",
        "CORTEX_KB_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _ns(**overrides: object) -> argparse.Namespace:
    """Build a Namespace with the KB-related fields the helpers read; per-test overrides win."""
    fields: dict[str, object] = {
        "local_kb_root": None,
        "cortex_kb_url": None,
        "degraded_kb":   False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)  # type: ignore[arg-type]


# _resolve_local_kb_root
def test_resolve_local_kb_root_uses_explicit_flag(
    env_clean: None, tmp_path: Path,
) -> None:
    """Highest-priority tier: ``--local-kb-root <path>`` wins."""
    args = _ns(local_kb_root=str(tmp_path / "from-flag"))
    assert _resolve_local_kb_root(args) == tmp_path / "from-flag"


def test_resolve_local_kb_root_falls_back_to_env(
    env_clean: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Tier 2: ``$HYPERLOOM_LOCAL_KB_ROOT`` when the flag is unset."""
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "from-env"))
    args = _ns()
    assert _resolve_local_kb_root(args) == tmp_path / "from-env"


def test_resolve_local_kb_root_falls_back_to_user_data_path(
    env_clean: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Tier 3: ``$USER_DATA_PATH/kb`` when neither flag nor HYPERLOOM_LOCAL_KB_ROOT is set."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    args = _ns()
    assert _resolve_local_kb_root(args) == tmp_path / "kb"


def test_resolve_local_kb_root_uses_workspace_default(
    env_clean: None,
) -> None:
    """Tier 4: the ``/workspace/hyperloom/kb`` last resort when no override is in scope."""
    args = _ns()
    assert _resolve_local_kb_root(args) == Path("/workspace/hyperloom/kb")


def test_resolve_local_kb_root_flag_beats_env(
    env_clean: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cross-tier precedence: flag wins when both flag and env are set."""
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "env"))
    args = _ns(local_kb_root=str(tmp_path / "flag"))
    assert _resolve_local_kb_root(args) == tmp_path / "flag"


def test_resolve_local_kb_root_does_not_create_directory(
    env_clean: None, tmp_path: Path,
) -> None:
    """Lazy creation: the helper only resolves the path; it does not create directories."""
    target = tmp_path / "lazy"
    args = _ns(local_kb_root=str(target))
    assert _resolve_local_kb_root(args) == target
    assert not target.exists()


# _build_recipe_kb_dispatcher
def test_build_dispatcher_returns_recipe_kb(
    env_clean: None, tmp_path: Path,
) -> None:
    args = _ns(local_kb_root=str(tmp_path), cortex_kb_url=None)
    kb = _build_recipe_kb_dispatcher(args)
    assert isinstance(kb, RecipeKB)
    assert isinstance(kb.local, LocalRecipeStore)
    assert kb.local.root == tmp_path


def test_build_dispatcher_no_remote_when_degraded_kb(
    env_clean: None, tmp_path: Path,
) -> None:
    """``--degraded-kb`` short-circuits remote regardless of any configured URL."""
    args = _ns(
        local_kb_root=str(tmp_path),
        cortex_kb_url="http://kb.example",
        degraded_kb=True,
    )
    kb = _build_recipe_kb_dispatcher(args)
    assert kb.remote is None


def test_build_dispatcher_no_remote_when_no_url(
    env_clean: None, tmp_path: Path,
) -> None:
    """No URL anywhere → local-only; the dispatcher wires ``remote=None``."""
    args = _ns(local_kb_root=str(tmp_path))
    kb = _build_recipe_kb_dispatcher(args)
    assert kb.remote is None


def test_build_dispatcher_wires_remote_when_url_passed(
    env_clean: None, tmp_path: Path,
) -> None:
    args = _ns(
        local_kb_root=str(tmp_path),
        cortex_kb_url="http://kb.example",
    )
    kb = _build_recipe_kb_dispatcher(args)
    assert isinstance(kb.remote, RemoteRecipeClient)
    assert kb.remote.kb_url == "http://kb.example"
    assert kb.remote.enabled is True


def test_build_dispatcher_wires_remote_from_env_url(
    env_clean: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``$CORTEX_KB_URL`` is the second-priority source, used when the flag is unset."""
    monkeypatch.setenv("CORTEX_KB_URL", "http://env-kb.example")
    args = _ns(local_kb_root=str(tmp_path))
    kb = _build_recipe_kb_dispatcher(args)
    assert isinstance(kb.remote, RemoteRecipeClient)
    assert kb.remote.kb_url == "http://env-kb.example"


def test_build_dispatcher_flag_url_beats_env(
    env_clean: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("CORTEX_KB_URL", "http://env-kb.example")
    args = _ns(
        local_kb_root=str(tmp_path),
        cortex_kb_url="http://flag-kb.example",
    )
    kb = _build_recipe_kb_dispatcher(args)
    assert kb.remote is not None
    assert kb.remote.kb_url == "http://flag-kb.example"


def test_build_dispatcher_uses_foreground_profile_for_remote(
    env_clean: None, tmp_path: Path,
) -> None:
    """The CLI always wires the foreground profile so a slow remote can't stall the main loop."""
    args = _ns(
        local_kb_root=str(tmp_path),
        cortex_kb_url="http://kb.example",
    )
    kb = _build_recipe_kb_dispatcher(args)
    assert kb.remote is not None
    assert kb.remote.foreground is True


def test_build_dispatcher_idempotent(
    env_clean: None, tmp_path: Path,
) -> None:
    """Two builds with the same args produce equivalent but distinct dispatchers (no shared mutable state)."""
    args = _ns(local_kb_root=str(tmp_path))
    a = _build_recipe_kb_dispatcher(args)
    b = _build_recipe_kb_dispatcher(args)
    assert a is not b
    assert a.local.root == b.local.root
    assert a.remote is None and b.remote is None


# Argparse parser integration — flag really is wired
def test_parser_accepts_local_kb_root_flag() -> None:
    """End-to-end: the parser accepts ``--local-kb-root`` and exposes it on the Namespace."""
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    ns = parser.parse_args([
        "optimize",
        "--local-kb-root", "/tmp/explicit",
        "--target-tput", "1.0",
    ])
    assert ns.local_kb_root == "/tmp/explicit"


def test_parser_default_local_kb_root_is_none() -> None:
    """``Namespace.local_kb_root`` defaults to ``None`` so the resolver's flag tier works."""
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    ns = parser.parse_args(["optimize", "--target-tput", "1.0"])
    assert ns.local_kb_root is None
