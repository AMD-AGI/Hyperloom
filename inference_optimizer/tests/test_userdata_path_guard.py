# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Guard tests for the ``$USER_DATA_PATH`` write-determinism contract.

When set, no resolver yields the pod-local default; when unset, the
fallback still works but emits a single loud warning. Covers
``paths.workspace_root`` and the CLI ``_resolve_local_kb_root``.
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
    """Wipe the resolver env vars and reset the one-shot warn guard for test isolation."""
    for key in (
        paths.ENV_USER_DATA_PATH,
        "HYPERLOOM_LOCAL_KB_ROOT",
        paths.ENV_CURRENT_SESSION_DIR,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(paths, "_WARNED_NO_USER_DATA", False)


def _args_without_overrides() -> argparse.Namespace:
    """A Namespace with no KB-root override so ``_resolve_local_kb_root`` exercises the env/default tiers."""
    return argparse.Namespace(local_kb_root=None)


# USER_DATA_PATH SET: never returns /workspace/hyperloom
def test_workspace_root_returns_user_data_path_when_set(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    assert paths.workspace_root() == tmp_path


def test_kb_root_under_user_data_path_when_set(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """With the env set, the local-KB root lands under it and never contains the pod-local default."""
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
    """The loud warning fires ONLY on misconfiguration; a correctly configured run stays quiet."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="inference_optimizer.paths"):
        paths.workspace_root()
    assert not [r for r in caplog.records if paths.ENV_USER_DATA_PATH in r.message]


# USER_DATA_PATH UNSET: loud fallback to /workspace/hyperloom
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
    """The KB-root fallback stays at ``/workspace/hyperloom/kb`` (now not silent)."""
    resolved = _resolve_local_kb_root(_args_without_overrides())
    assert resolved == Path(_DEFAULT) / "kb"


def test_unset_warning_is_emitted_once(
    clean_env: None, caplog: pytest.LogCaptureFixture,
) -> None:
    """Hot-path guard: the warning fires at most once per process so it doesn't drown logs."""
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


# Manifest dependency provenance: out-of-tree runtime overrides are loud
def test_manifest_warns_when_dependency_is_pod_local(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A dependency override into a pod-local /workspace path must warn loudly."""
    from inference_optimizer import manifest

    user_data = tmp_path / "user_data"
    # The dir need not exist; the escape guard runs before the is_dir gate.
    pod_local = "/workspace/hyperloom_runtime_smoke/Magpie"
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(user_data))
    monkeypatch.setenv("MAGPIE_DIR", pod_local)

    with caplog.at_level(logging.WARNING, logger="inference_optimizer.manifest"):
        manifest._describe_dep("MAGPIE_DIR")

    assert any(
        "MAGPIE_DIR" in r.message and "pod-local" in r.message
        for r in caplog.records
    ), "expected a loud pod-local escape warning"


def test_manifest_does_not_warn_for_persistent_shared_checkout(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A persistent shared checkout outside USER_DATA_PATH (e.g. a WekaFS mirror) is legitimate and must NOT warn."""
    from inference_optimizer import manifest

    user_data = tmp_path / "user_data"
    user_data.mkdir(parents=True)
    # A WekaFS mirror outside USER_DATA_PATH (not under a pod-local prefix); need not exist on disk.
    shared = "/wekafs/shared-mirrors/InferenceX"
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(user_data))
    monkeypatch.setenv("INFERENCEX_PATH", shared)

    with caplog.at_level(logging.WARNING, logger="inference_optimizer.manifest"):
        dep = manifest._describe_dep("INFERENCEX_PATH")

    assert dep["path"] == shared
    assert not [r for r in caplog.records if "INFERENCEX_PATH" in r.message]


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
