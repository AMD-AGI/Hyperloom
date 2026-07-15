# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``resolve_local_model_dir`` (repo id / path -> local model dir).

Covers the shared resolver that lets in-process config readers (roofline
ceiling, model-config summary, KB tags, model-class inference, fp8 detection)
accept a bare HF repo id, not only a local directory.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from hyperloom.inference_optimizer.model_config_utils import (
    _load_model_config_dict,
    resolve_local_model_dir,
)


def _make_model_dir(tmp_path: Path, cfg: dict | None = None) -> Path:
    """Create a fake local model dir carrying a ``config.json``."""
    d = tmp_path / "weights"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(cfg or {"model_type": "llama"}), encoding="utf-8"
    )
    return d


def test_resolve_empty_returns_none():
    assert resolve_local_model_dir("") is None
    assert resolve_local_model_dir(None) is None
    assert resolve_local_model_dir("   ") is None


def test_resolve_existing_dir_returned_as_is(tmp_path):
    d = _make_model_dir(tmp_path)
    assert resolve_local_model_dir(str(d)) == d
    assert resolve_local_model_dir(d) == d


def test_resolve_repo_id_uncached_returns_none(monkeypatch):
    # huggingface_hub missing (or a cache miss) -> graceful None. Forcing the
    # module to None in sys.modules makes the import raise deterministically.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert resolve_local_model_dir("Qwen/Qwen3-0.6B") is None


def test_resolve_repo_id_hits_hf_cache(tmp_path, monkeypatch):
    d = _make_model_dir(tmp_path)
    cfg_path = str(d / "config.json")
    seen: dict = {}

    fake = types.ModuleType("huggingface_hub")

    def _try(repo_id, filename, **kw):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        return cfg_path

    fake.try_to_load_from_cache = _try  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    assert resolve_local_model_dir("Qwen/Qwen3-0.6B") == d
    assert seen == {"repo_id": "Qwen/Qwen3-0.6B", "filename": "config.json"}


def test_resolve_repo_id_cache_sentinel_returns_none(monkeypatch):
    # A known-absent entry returns a non-str sentinel object -> None.
    fake = types.ModuleType("huggingface_hub")
    fake.try_to_load_from_cache = lambda repo_id, filename, **kw: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert resolve_local_model_dir("org/absent") is None


def test_load_model_config_dict_uses_resolver(tmp_path, monkeypatch):
    d = _make_model_dir(tmp_path, {"model_type": "qwen3"})
    cfg_path = str(d / "config.json")

    fake = types.ModuleType("huggingface_hub")
    fake.try_to_load_from_cache = lambda repo_id, filename, **kw: cfg_path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    # A bare repo id (not a local dir) now resolves via the HF cache and parses.
    data = _load_model_config_dict("Qwen/Qwen3-8B")
    assert isinstance(data, dict)
    assert data.get("model_type") == "qwen3"


def test_load_model_config_dict_local_dir_unchanged(tmp_path):
    # Regression: an existing local dir still parses without any resolver detour.
    d = _make_model_dir(tmp_path, {"model_type": "mixtral"})
    data = _load_model_config_dict(str(d))
    assert isinstance(data, dict)
    assert data.get("model_type") == "mixtral"
