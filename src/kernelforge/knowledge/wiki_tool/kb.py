# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a knowledge base checkout's scripts need to search a Hyperloom Experience KB beside it.

InstinctWiki's scripts read the KB from their environment (its README, "Experience KB search"). The KB is the one the
user configured, a workspace's local KB by default; the schema is the one KernelForge's own records are written under
unless the user names another; the context names the kernel this campaign optimizes. The token reaches the scripts as
a file the process alone can read, never as a value: a server's environment is handed to the agent CLI on its command
line.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from hyperloom_kb.workspace import TOKEN_ENV, TOKEN_FILE_ENV, URL_ENV

from kernelforge.knowledge.kb_store.identity.run import campaign_identity
from hyperloom_common.io import DURABLE, atomic_write_text

SCHEMA_ENV = "HYPERLOOM_KB_SCHEMA"
CONTEXT_ENV = "HYPERLOOM_KB_CONTEXT"
CACHE_ENV = "HYPERLOOM_KB_CACHE"
CONTEXT_NAME = "knowledge_wiki_kb_context.json"
KERNELFORGE_MAPPING = "kernelforge-v1"


def write_kb_context(experiments_dir: Path, campaign: Mapping[str, Any]) -> Path:
    """Name the kernel this campaign optimizes, from its campaign_config.json, in the schema's identity fields."""
    path = Path(experiments_dir) / CONTEXT_NAME
    atomic_write_text(
        path, json.dumps({"identity": campaign_identity(campaign)}, indent=2, sort_keys=True) + "\n", options=DURABLE
    )
    return path


@lru_cache(maxsize=1)
def kernelforge_schema_ref() -> str:
    """The schema KernelForge's records are collected under, by its packaged mapping."""
    from hyperloom_kb.collect import load_mapping

    return load_mapping(KERNELFORGE_MAPPING).declaration.schema_ref


@lru_cache(maxsize=None)
def _token_file(token: str) -> Path:
    """The token in a file of a private directory this process removes when it exits."""
    directory = Path(tempfile.mkdtemp(prefix="kernelforge-kb-"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    path = directory / "token"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token)
    return path


def kb_environment(context_path: Path | None, environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """The KB settings for the scripts' environment; none without a configured KB URL and token."""
    url, token = environ.get(URL_ENV, "").strip(), environ.get(TOKEN_ENV, "").strip()
    token_file = str(_token_file(token)) if token else environ.get(TOKEN_FILE_ENV, "").strip()
    if not url or not token_file:
        return {}
    env = {
        URL_ENV: url,
        TOKEN_FILE_ENV: token_file,
        SCHEMA_ENV: environ.get(SCHEMA_ENV, "").strip() or kernelforge_schema_ref(),
    }
    if environ.get(CACHE_ENV, "").strip():
        env[CACHE_ENV] = environ[CACHE_ENV].strip()
    if context_path is not None and Path(context_path).is_file():
        env[CONTEXT_ENV] = str(context_path)
    return env
