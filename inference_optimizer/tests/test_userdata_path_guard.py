"""Guard tests for the ``$USER_DATA_PATH`` write-determinism contract.

These lock the "写死" (make-it-deterministic) requirement:

1. When ``$USER_DATA_PATH`` IS set, no resolver may ever yield the
   pod-local ``/workspace/hyperloom`` default — every artefact path is
   anchored on the operator-chosen root.
2. When ``$USER_DATA_PATH`` is NOT set, the resolver still falls back to
   ``/workspace/hyperloom`` (so a degraded sandbox keeps working) BUT
   emits a single loud ``logging.warning`` so the misconfiguration is
   never silent.

Covers ``inference_optimizer.paths.workspace_root`` and the CLI helper
``_resolve_local_kb_root`` that builds on it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest

from inference_optimizer import paths
from inference_optimizer.cli import _resolve_local_kb_root

_DEFAULT = "/workspace/hyperloom"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe every env var the resolvers consult and reset the one-shot
    warn guard so each test is fully isolated from leakage by an earlier
    test in the same process."""
    for key in (
        paths.ENV_USER_DATA_PATH,
        "HYPERLOOM_LOCAL_KB_ROOT",
        paths.ENV_CURRENT_SESSION_DIR,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths, "_WARNED_NO_USER_DATA", False)


def _args_without_overrides() -> argparse.Namespace:
    """A Namespace with no KB-root override, matching the argparse
    default so ``_resolve_local_kb_root`` exercises the env/default
    tiers rather than the explicit flag."""
    return argparse.Namespace(local_kb_root=None)


# ---------------------------------------------------------------------------
# USER_DATA_PATH SET: never returns /workspace/hyperloom
# ---------------------------------------------------------------------------
def test_workspace_root_returns_user_data_path_when_set(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    assert paths.workspace_root() == tmp_path


def test_kb_root_under_user_data_path_when_set(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """With the env set, the local-KB root lands under it and the
    pod-local default literal never appears anywhere in the path."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    resolved = _resolve_local_kb_root(_args_without_overrides())
    assert resolved == tmp_path / "kb"
    assert tmp_path in resolved.parents
    assert _DEFAULT not in str(resolved)


def test_no_warning_emitted_when_set(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The loud warning must fire ONLY on misconfiguration — a correctly
    configured run stays quiet."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="inference_optimizer.paths"):
        paths.workspace_root()
    assert not [r for r in caplog.records if paths.ENV_USER_DATA_PATH in r.message]


# ---------------------------------------------------------------------------
# USER_DATA_PATH UNSET: loud fallback to /workspace/hyperloom
# ---------------------------------------------------------------------------
def test_workspace_root_falls_back_and_warns_when_unset(
    clean_env: None, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="inference_optimizer.paths"):
        result = paths.workspace_root()
    assert result == paths.DEFAULT_SESSION_DIR == Path(_DEFAULT)
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and paths.ENV_USER_DATA_PATH in r.message
    ]
    assert warnings, "expected a loud USER_DATA_PATH-unset warning"


def test_kb_root_falls_back_when_unset(clean_env: None) -> None:
    """The KB-root fallback stays at ``/workspace/hyperloom/kb`` (degraded
    sandbox keeps working) — unchanged value, now just not silent."""
    resolved = _resolve_local_kb_root(_args_without_overrides())
    assert resolved == Path(_DEFAULT) / "kb"


def test_unset_warning_is_emitted_once(
    clean_env: None, caplog: pytest.LogCaptureFixture,
) -> None:
    """Hot-path guard: many call sites route through workspace_root(); the
    warning must fire at most once per process so it does not drown logs."""
    with caplog.at_level(logging.WARNING, logger="inference_optimizer.paths"):
        paths.workspace_root()
        paths.workspace_root()
        paths.workspace_root()
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and paths.ENV_USER_DATA_PATH in r.message
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Manifest dependency provenance: out-of-tree runtime overrides are loud
# ---------------------------------------------------------------------------
def test_manifest_warns_when_dependency_escapes_user_data(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from inference_optimizer import manifest

    user_data = tmp_path / "user_data"
    outside = tmp_path / "outside_runtime" / "Magpie"
    outside.mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(user_data))
    monkeypatch.setenv("MAGPIE_DIR", str(outside))

    with caplog.at_level(logging.WARNING, logger="inference_optimizer.manifest"):
        dep = manifest._describe_dep("MAGPIE_DIR")

    assert dep["path"] == str(outside)
    assert any(
        "MAGPIE_DIR" in r.message
        and "outside USER_DATA_PATH" in r.message
        for r in caplog.records
    )


def test_manifest_does_not_warn_when_dependency_under_user_data(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from inference_optimizer import manifest

    user_data = tmp_path / "user_data"
    magpie = user_data / "runtime" / "Magpie"
    magpie.mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(user_data))
    monkeypatch.setenv("MAGPIE_DIR", str(magpie))

    with caplog.at_level(logging.WARNING, logger="inference_optimizer.manifest"):
        dep = manifest._describe_dep("MAGPIE_DIR")

    assert dep["path"] == str(magpie)
    assert not [r for r in caplog.records if "MAGPIE_DIR" in r.message]
