# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tool cache holding the InferenceX client inference_suite stages, at its pin, as a staging that verifies."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import TOOL_CACHE_ENV
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT, INFERENCEX_TOOL
from inference_suite.staging import INFERENCEX_REQUIRED, MANIFEST_FILE, StagedTool, StagingManifest, tool_root


def verified_tool_cache(tool_cache: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stage the pinned client's required files in ``tool_cache`` and name it the application's tool cache; returns
    the staged InferenceX root."""

    root = tool_root(tool_cache, INFERENCEX_TOOL, INFERENCEX_CLIENT_COMMIT)
    files = {}
    for relative in INFERENCEX_REQUIRED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = StagingManifest({INFERENCEX_TOOL: StagedTool(INFERENCEX_CLIENT_COMMIT, files)})
    (tool_cache / MANIFEST_FILE).write_bytes(manifest.to_bytes())
    monkeypatch.setenv(TOOL_CACHE_ENV, str(tool_cache))
    return root
