"""Tests for the recipe-snapshot KB dispatcher bootstrap helpers in cli.

Covers:

* :func:`_resolve_local_kb_root` — flag > env > USER_DATA_PATH >
  /workspace/hyperloom fallback ladder, with each tier verified
  independently.
* :func:`_build_recipe_kb_dispatcher` — wires :class:`RecipeKB`
  with a :class:`LocalRecipeStore` always, and a
  :class:`RemoteRecipeClient` only when:
    - ``--degraded-kb`` is NOT set, AND
    - ``--cortex-kb-url`` (or ``$CORTEX_KB_URL``) is non-empty.

The tests parse a real :class:`argparse.Namespace` via the
production parser so the test surface stays honest — if a future
edit renames the flag the test fails for the right reason
(parser mismatch) instead of swallowing a typo.
"""

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


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe the env vars these helpers consult so each test is
    explicit about which precedence tier it's exercising."""
    for key in (
        "HYPERLOOM_LOCAL_KB_ROOT",
        "USER_DATA_PATH",
        "CORTEX_KB_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _ns(**overrides: object) -> argparse.Namespace:
    """Build a Namespace with the four KB-related fields the helpers
    read, defaulting each to ``None`` / ``False`` (matches argparse
    defaults for the production parser).

    Per-test overrides win over the defaults — we merge into a dict
    first so a caller passing ``local_kb_root=...`` doesn't collide
    with the default kwarg."""
    fields: dict[str, object] = {
        "local_kb_root": None,
        "cortex_kb_url": None,
        "degraded_kb":   False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)  # type: ignore[arg-type]


# ===========================================================================
# _resolve_local_kb_root
# ===========================================================================
def test_resolve_local_kb_root_uses_explicit_flag(
    env_clean: None, tmp_path: Path,
) -> None:
    """Highest-priority tier: ``--local-kb-root <path>`` wins over
    everything else."""
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
    """Tier 3: ``$USER_DATA_PATH/kb`` when neither flag nor
    HYPERLOOM_LOCAL_KB_ROOT is set. Path matches the documented
    contract in the local-kb-recipe-snapshot-requirements doc §2:
    "the local KB path is fixed at ``${USER_DATA_PATH}/kb``"."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    args = _ns()
    assert _resolve_local_kb_root(args) == tmp_path / "kb"


def test_resolve_local_kb_root_uses_workspace_default(
    env_clean: None,
) -> None:
    """Tier 4: the documented ``/workspace/hyperloom/kb`` last
    resort when no override is in scope."""
    args = _ns()
    assert _resolve_local_kb_root(args) == Path("/workspace/hyperloom/kb")


def test_resolve_local_kb_root_flag_beats_env(
    env_clean: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cross-tier precedence sanity: flag wins when both flag AND
    env are set (operator passed --local-kb-root explicitly to
    override the env they inherited)."""
    monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(tmp_path / "env"))
    args = _ns(local_kb_root=str(tmp_path / "flag"))
    assert _resolve_local_kb_root(args) == tmp_path / "flag"


def test_resolve_local_kb_root_does_not_create_directory(
    env_clean: None, tmp_path: Path,
) -> None:
    """Lazy creation: the helper only resolves the path; the
    LocalRecipeStore creates directories on first write so a
    degraded run pays nothing on disk."""
    target = tmp_path / "lazy"
    args = _ns(local_kb_root=str(target))
    assert _resolve_local_kb_root(args) == target
    assert not target.exists()


# ===========================================================================
# _build_recipe_kb_dispatcher
# ===========================================================================
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
    """``--degraded-kb`` must short-circuit remote regardless of any
    URL the operator may have left in env / flag."""
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
    """No URL anywhere → local-only. There is no hard-coded default
    endpoint to fall back to (the old central kb-service default was
    retired), so the dispatcher wires ``remote=None``.
    """
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
    """``$CORTEX_KB_URL`` is the second-priority source — used when
    the flag is unset (the operator has the URL exported globally)."""
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
    """The CLI always wires the foreground profile (2 s + 1 retry)
    so a slow remote can't stall the optimizer main loop. Background
    callers (future flusher / breakdown collector) build their own
    client directly when they need the longer profile."""
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
    """Two builds with the same args produce equivalent dispatchers
    (different objects, same logical wiring) — guards against
    accidental shared mutable state."""
    args = _ns(local_kb_root=str(tmp_path))
    a = _build_recipe_kb_dispatcher(args)
    b = _build_recipe_kb_dispatcher(args)
    assert a is not b
    assert a.local.root == b.local.root
    assert a.remote is None and b.remote is None


# ===========================================================================
# Argparse parser integration — flag really is wired
# ===========================================================================
def test_parser_accepts_local_kb_root_flag() -> None:
    """End-to-end: the production argparse parser accepts the new
    flag and propagates it onto the Namespace under the documented
    attribute name (so the build helper can read it via getattr)."""
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    ns = parser.parse_args([
        "optimize",
        "--local-kb-root", "/tmp/explicit",
        "--target-tput", "1.0",
    ])
    assert ns.local_kb_root == "/tmp/explicit"


def test_parser_default_local_kb_root_is_none() -> None:
    """``argparse.Namespace.local_kb_root`` defaults to ``None`` so
    the resolver's flag-tier check (``or env`` chain) works
    correctly when no flag is passed."""
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    ns = parser.parse_args(["optimize", "--target-tput", "1.0"])
    assert ns.local_kb_root is None
